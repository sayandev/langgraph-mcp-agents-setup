import streamlit as st
import asyncio
import nest_asyncio
import json
import os
import platform
import re 

if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Apply nest_asyncio: Allow nested calls within an already running event loop
nest_asyncio.apply()

# Create and reuse global event loop (create once and continue using)
if "event_loop" not in st.session_state:
    loop = asyncio.new_event_loop()
    st.session_state.event_loop = loop
    asyncio.set_event_loop(loop)

from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from utils import astream_graph, random_uuid
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.messages.tool import ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

# Load environment variables (get API keys and settings from .env file)
load_dotenv(override=True)

# config.json file path setting
CONFIG_FILE_PATH = "config.json"

# Function to load settings from JSON file
def load_config_from_json():
    default_config = {
  "get_current_time": {
    "command": "python",
    "args": [
       // To Do: Import mcp_server_time
    ],
    "transport": "stdio"
  },
"CustomerSupportAssistantTools": {
    "command": "python",
    "args": [
       // To Do: Import mcp_server_remote
    ],
    "transport": "stdio"
  }
}
    try:
        if os.path.exists(CONFIG_FILE_PATH):
            with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            save_config_to_json(default_config)
            return default_config
    except Exception as e:
        st.error(f"Error loading settings file: {str(e)}")
        return default_config

# Function to save settings to JSON file
def save_config_to_json(config):
    try:
        with open(CONFIG_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"Error saving settings file: {str(e)}")
        return False

# Initialize login session variables
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

use_login = os.environ.get("USE_LOGIN", "false").lower() == "true"

image_path = "assets/Zwigato_icon.png"

if use_login and not st.session_state.authenticated:
    st.set_page_config(page_title="Zwigato Support Agent", page_icon=image_path)
else:
    st.set_page_config(page_title="Zwigato Support Agent", page_icon=image_path, layout="wide")
    # Inject CSS for chat borders
    st.markdown(
        """
        <style>
        /* Chat message container border */
        [data-testid="stChatMessage"] {
            border: 1px solid #bbbbbb !important;
            border-radius: 8px !important;
            padding: 8px !important;
            background-color: #ffffff !important;
        }
        /* Chat input box border */
        [data-testid="stChatInput"] {
            border: 1px solid #bbbbbb !important;
            border-radius: 8px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

if use_login and not st.session_state.authenticated:
    st.title("🔐 Login")
    st.markdown("Login is required to use the system.")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")
        if submit_button:
            expected_username = os.environ.get("USER_ID")
            expected_password = os.environ.get("USER_PASSWORD")
            if username == expected_username and password == expected_password:
                st.session_state.authenticated = True
                st.success("✅ Login successful! Please wait...")
                st.rerun()
            else:
                st.error("❌ Username or password is incorrect.")
    st.stop()

st.sidebar.divider()
st.sidebar.image(image_path, use_container_width=True, caption="Zwigato Customer Support Agent")
st.image(image_path, use_container_width=False,width=100 )
st.title("Zwigato Customer Support Agent")
st.markdown("✨ How can I assist you today?")

SYSTEM_PROMPT = """<ROLE>
You are a smart agent with an ability to use tools. 
You will be given a question and you will use the tools to answer the question.
Pick the most relevant tool to answer the question. 
If you are failed to answer the question, try different tools to get context.
Your answer should be very polite and professional.
</ROLE>
----
<INSTRUCTIONS>
Step 1: Analyze the question
- Analyze user's question and final goal.
- If the user's question is consist of multiple sub-questions, split them into smaller sub-questions.

Step 2: Pick the most relevant tool
- Pick the most relevant tool to answer the question.
- If you are failed to answer the question, try different tools to get context.

Step 3: Answer the question
- Answer the question in the same language as the question.
- Your answer should be very polite and professional.

Step 4: Provide the source of the answer(if applicable)
- If you've used the tool, provide the source of the answer.
- Valid sources are either a website(URL) or a document(PDF, etc).

Guidelines:
- If you've used the tool, your answer should be based on the tool's output(tool's output is more important than your own knowledge).
- If you've used the tool, and the source is valid URL, provide the source(URL) of the answer.
- Skip providing the source if the source is not URL.
- Answer in the same language as the question.
- Answer should be concise and to the point.
- Avoid response your output with any other information than the answer and the source.  
</INSTRUCTIONS>
----
<OUTPUT_FORMAT>
(concise answer to the question)


</OUTPUT_FORMAT>
"""

# **Source**(if applicable)
# - (source1: valid URL)
# - (source2: valid URL)
# - ...

OUTPUT_TOKEN_INFO = {
    "gpt-4o-mini": {"max_tokens": 16000},
    "claude-3-5-sonnet-latest": {"max_tokens": 8192},
    "claude-3-5-haiku-latest": {"max_tokens": 8192},
    "claude-3-7-sonnet-latest": {"max_tokens": 64000},
    "gpt-4o": {"max_tokens": 16000},
}

if "session_initialized" not in st.session_state:
    st.session_state.session_initialized = False
    st.session_state.agent = None
    st.session_state.history = []
    st.session_state.mcp_client = None
    st.session_state.timeout_seconds = 120
    st.session_state.selected_model = "gpt-4o-mini"
    st.session_state.recursion_limit = 100

if "thread_id" not in st.session_state:
    st.session_state.thread_id = random_uuid()

async def cleanup_mcp_client():
    if "mcp_client" in st.session_state and st.session_state.mcp_client is not None:
        try:
            await st.session_state.mcp_client.__aexit__(None, None, None)
            st.session_state.mcp_client = None
        except Exception:
            pass

def print_message():
    i = 0
    while i < len(st.session_state.history):
        message = st.session_state.history[i]
        if message["role"] == "user":
            st.chat_message("user", avatar="🧑‍💻").markdown(message["content"])
            i += 1
        elif message["role"] == "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(message["content"]) 
                if (i + 1 < len(st.session_state.history) and
                        st.session_state.history[i + 1]["role"] == "assistant_tool"):
                    with st.expander("🔧 Tool Call Information", expanded=False):
                        st.markdown(st.session_state.history[i + 1]["content"])
                    i += 2
                else:
                    i += 1
        else:
            i += 1

def normalize_for_log(text_content: str) -> str:
    if not text_content:
        return ""
    text_content = re.sub(r'\s+', ' ', text_content)
    text_content = text_content.strip()
    return text_content

def get_streaming_callback(text_placeholder, tool_placeholder):
    accumulated_text_list = []
    accumulated_tool_log_list = []
    current_tool_call_args_buffer = {}
    # Track seen tool call invocations to avoid duplicate logging
    seen_raw_jsons = set()

    def callback_func(event_data: dict): 
        nonlocal accumulated_text_list, accumulated_tool_log_list, current_tool_call_args_buffer
        
        print("[CALLBACK_START] Received event_data:", json.dumps(event_data, indent=2, default=str))
        
        message_content = None
        if isinstance(event_data, dict):
            if 'data' in event_data and isinstance(event_data['data'], dict) and 'chunk' in event_data['data']:
                message_content = event_data['data']['chunk']
                print("[CALLBACK_DATA] Found chunk in event_data['data']")
            elif 'chunk' in event_data:
                message_content = event_data['chunk']
                print("[CALLBACK_DATA] Found chunk in event_data")
            elif 'content' in event_data and isinstance(event_data['content'], (AIMessageChunk, ToolMessage)):
                message_content = event_data.get("content")
                print("[CALLBACK_DATA] Found content in event_data")
        elif isinstance(event_data, (AIMessageChunk, ToolMessage)):
            message_content = event_data
            print("[CALLBACK_DATA] event_data is AIMessageChunk or ToolMessage")
        
        if not message_content:
            print("[CALLBACK_ERROR] No message_content found")
            return None

        print("[CALLBACK_MESSAGE] Processing message_content type:", type(message_content).__name__)

        if isinstance(message_content, AIMessageChunk):
            print(f"[CALLBACK_RAW_CHUNK_CONTENT] Raw AIMessageChunk content: {message_content.content!r}")
            
            text_from_chunk = ""
            if isinstance(message_content.content, str) and message_content.content.strip():
                text_from_chunk = message_content.content
            elif isinstance(message_content.content, list):
                for block in message_content.content:
                    if block.get("type") == "text" and block.get("text","").strip():
                        text_from_chunk += block.get("text","")
            
            print(f"[CALLBACK_TEXT_FROM_CHUNK] Extracted text_from_chunk: {text_from_chunk!r}")          
            if text_from_chunk: 
                # Only add space between chunks if really needed (no existing space or newline)
                if accumulated_text_list and accumulated_text_list[-1].strip() and text_from_chunk.strip():
                    last_char = accumulated_text_list[-1][-1] if accumulated_text_list[-1] else ""
                    first_char = text_from_chunk[0] if text_from_chunk else ""
                accumulated_text_list.append(text_from_chunk)
                current_display_text = "".join(accumulated_text_list)
                print(f"[CALLBACK_ACCUMULATED_TEXT_UPDATE] Updating text_placeholder with: {current_display_text!r}")
                text_placeholder.markdown(current_display_text)

            raw_tool_call_json_for_expander_log_this_chunk = []
            
            if hasattr(message_content, "tool_call_chunks") and message_content.tool_call_chunks:
                print("[CALLBACK_TOOL_CALLS] Processing tool_call_chunks...")
                for tc_chunk in message_content.tool_call_chunks:
                    print(f"[CALLBACK_TOOL_CALL_CHUNK] Processing chunk: {json.dumps(tc_chunk, default=str)}")
                    # Skip if the tool name is null
                    if tc_chunk.get("name") is None:
                        print("[CALLBACK_TOOL_CALL_CHUNK] Skipping chunk due to null name")
                        continue

                    # Track the tool call args
                    if tc_chunk.get("args"):
                        tool_call_id = tc_chunk.get("id")
                        if tool_call_id not in current_tool_call_args_buffer:
                            current_tool_call_args_buffer[tool_call_id] = ""
                        print(f"[CALLBACK_TOOL_CALL_ARGS] Updating args buffer for tool {tc_chunk.get('name')} (ID: {tool_call_id})")
                        print(f"[CALLBACK_TOOL_CALL_ARGS] Current args: {tc_chunk.get('args')}")
                        current_tool_call_args_buffer[tool_call_id] += tc_chunk.get("args", "")
                        print(f"[CALLBACK_TOOL_CALL_ARGS] Updated buffer: {current_tool_call_args_buffer[tool_call_id]}")

                    # Prepare raw JSON for logging
                    raw_json = json.dumps(
                        {"name": tc_chunk.get("name"), "args_chunk": tc_chunk.get("args"), "id": tc_chunk.get("id"), "index": tc_chunk.get("index")},
                        indent=2, ensure_ascii=False
                    )
                    # Only log each invocation once
                    if raw_json not in seen_raw_jsons:
                        seen_raw_jsons.add(raw_json)
                        raw_tool_call_json_for_expander_log_this_chunk.append(raw_json)
                        print(f"[CALLBACK_TOOL_CALL_LOG] Added new tool call to log: {raw_json}")
                    
                    # Display action text only once per tool call
                    if tc_chunk.get("name") and tc_chunk.get("id") not in current_tool_call_args_buffer:
                        tool_call_id = tc_chunk.get("id")
                        current_tool_call_args_buffer[tool_call_id] = ""
                        action_text1 = f"""Thinking how to solve the customer query ... """
                        action_text2 = f"""\n performing the following **Action:** `{tc_chunk['name']}`"""
                        print(f"[CALLBACK_TOOL_CALL_ACTION] Displaying action for tool: {tc_chunk['name']}")
                        if accumulated_text_list and accumulated_text_list[-1].strip() and not accumulated_text_list[-1].endswith("\n\n"):
                            accumulated_text_list.append("\n\n")
                        accumulated_text_list.append(action_text1)
                        accumulated_text_list.append(action_text2)
                        text_placeholder.markdown("".join(accumulated_text_list))

            # Handle tool_use blocks in message content
            if isinstance(message_content.content, list):
                for block in message_content.content:
                    if block.get("type") == "tool_use":
                        print(f"[CALLBACK_TOOL_USE_BLOCK] Processing tool_use block: {json.dumps(block, default=str)}")
                        if block.get("name") is None:
                            print("[CALLBACK_TOOL_USE_BLOCK] Skipping block due to null name")
                            continue
                            
                        tool_name = block.get("name", "UnknownTool")
                        tool_input = block.get("input", {})
                        print(f"[CALLBACK_TOOL_USE_INPUT] Tool {tool_name} input: {json.dumps(tool_input, default=str)}")
                        tool_input_str = json.dumps(tool_input, indent=2, ensure_ascii=False)

                        action_text = f"**Action:** `{tool_name}`\n**Action Input:**\n```json\n{tool_input_str}\n```"
                        if accumulated_text_list and accumulated_text_list[-1].strip() and not accumulated_text_list[-1].endswith("\n\n"):
                            accumulated_text_list.append("\n\n")
                        elif not accumulated_text_list:
                            pass
                        accumulated_text_list.append(action_text)
                        text_placeholder.markdown("".join(accumulated_text_list))
                        
                        raw_tool_call_json_for_expander_log_this_chunk.append(json.dumps(block, indent=2, ensure_ascii=False))
            
            # Log tool calls in expander
            if raw_tool_call_json_for_expander_log_this_chunk: 
                print("[CALLBACK_TOOL_CALLS_SUMMARY] Adding tool calls to expander")
                for raw_json in raw_tool_call_json_for_expander_log_this_chunk:
                    accumulated_tool_log_list.append(f"\n**Tool Call Invocation:**\n```json\n{raw_json}\n```\n")
                with tool_placeholder.expander("🔧 Tool Call Information", expanded=True):
                    st.markdown("".join(accumulated_tool_log_list))

        elif isinstance(message_content, ToolMessage):
            print(f"[CALLBACK_TOOL_MESSAGE] Processing ToolMessage: {json.dumps(message_content.content, default=str)}")
            tool_name = getattr(message_content, 'name', 'unknown_tool_observation') 
            observation_content_str = ""
            if isinstance(message_content.content, str):
                observation_content_str = message_content.content
            else: 
                try:
                    observation_content_str = json.dumps(message_content.content, indent=2, ensure_ascii=False)
                except TypeError:
                    observation_content_str = str(message_content.content)
              
            # Get the tool input from the args_buffer if available
            tool_input = None
            for buffer_id, buffer_content in current_tool_call_args_buffer.items():
                if buffer_content.strip():
                    try:
                        tool_input = json.loads(buffer_content)
                        print(f"[CALLBACK_TOOL_MESSAGE_INPUT] Found tool input from buffer {buffer_id}: {json.dumps(tool_input, default=str)}")
                    except:
                        tool_input = buffer_content
                        print(f"[CALLBACK_TOOL_MESSAGE_INPUT] Found non-JSON tool input from buffer {buffer_id}: {tool_input}")

            # Add Tool Input section if available
            if tool_input:
                accumulated_tool_log_list.append(f"\n**Tool Input:**\n```json\n{json.dumps(tool_input, indent=2)}\n```\n")

            # Add Tool Output section
            normalized_observation_for_log = normalize_for_log(observation_content_str)
            print(f"[CALLBACK_TOOL_MESSAGE_OUTPUT] Tool {tool_name} output: {normalized_observation_for_log}")
            accumulated_tool_log_list.append(f"\n**Tool Output (from tool: {tool_name}):**\n```json\n{normalized_observation_for_log}\n```\n")
            
            with tool_placeholder.expander("🔧 Tool Call Information", expanded=True):
                st.markdown("".join(accumulated_tool_log_list))

            if accumulated_text_list and accumulated_text_list[-1].strip() and not accumulated_text_list[-1].endswith("\n\n"):
                accumulated_text_list.append("\n\n")
            current_display_text_after_tool_msg = "".join(accumulated_text_list)
            print(f"[CALLBACK_ACCUMULATED_TEXT_UPDATE] Updating text_placeholder after ToolMessage: {current_display_text_after_tool_msg!r}")
            text_placeholder.markdown(current_display_text_after_tool_msg)
        return None
    return callback_func, accumulated_text_list, accumulated_tool_log_list


async def process_query(query, text_placeholder, tool_placeholder, timeout_seconds=60):
    verbose_streaming_callback, accumulated_text_obj, accumulated_tool_log_obj = \
        get_streaming_callback(text_placeholder, tool_placeholder)
    response_from_astream_graph = None
    try:
        if st.session_state.agent:
            try:
                response_from_astream_graph = await asyncio.wait_for(
                    astream_graph(
                        st.session_state.agent,
                        {"messages": [HumanMessage(content=query)]},
                        callback=verbose_streaming_callback,
                        config=RunnableConfig(
                            recursion_limit=st.session_state.recursion_limit,
                            thread_id=st.session_state.thread_id,
                        ),
                    ),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                error_msg = f"⏱️ Request time exceeded {timeout_seconds} seconds. Please try again later."
                final_text_on_timeout = "".join(accumulated_text_obj)
                final_tool_log_on_timeout = "".join(accumulated_tool_log_obj)
                print(f"[PROCESS_QUERY_TIMEOUT] Final text on timeout: {final_text_on_timeout!r}")
                print(f"[PROCESS_QUERY_TIMEOUT] Final tool log on timeout: {final_tool_log_on_timeout!r}")
                return {"error": error_msg}, final_text_on_timeout, final_tool_log_on_timeout
            except Exception as e_inner:
                import traceback
                error_msg_inner = f"❌ Error during agent execution: {str(e_inner)}\n{traceback.format_exc()}"
                final_text_on_error = "".join(accumulated_text_obj)
                final_tool_log_on_error = "".join(accumulated_tool_log_obj)
                print(f"[PROCESS_QUERY_AGENT_ERROR] Final text on agent error: {final_text_on_error!r}")
                print(f"[PROCESS_QUERY_AGENT_ERROR] Final tool log on agent error: {final_tool_log_on_error!r}")
                return {"error": error_msg_inner}, final_text_on_error, final_tool_log_on_error

            final_text = "".join(accumulated_text_obj)
              # Just fix the Zwigato name if needed
            if "Z wig ato" in final_text:
                print(f"[PROCESS_QUERY_INFO] 'Z wig ato' found in final_text. Replacing with 'Zwigato'.")
                final_text = final_text.replace("Z wig ato", "Zwigato")

            final_tool_log = "".join(accumulated_tool_log_obj)
            print(f"[PROCESS_QUERY_FINAL_TEXT] Final text output: {final_text!r}")
            print(f"[PROCESS_QUERY_FINAL_TOOL_LOG] Final tool log output: {final_tool_log!r}")
            return response_from_astream_graph, final_text, final_tool_log
        else:
            return {"error": "🚫 Agent has not been initialized."}, "🚫 Agent has not been initialized.", ""
    except Exception as e_outer:
        import traceback
        error_msg_outer = f"❌ Unexpected error in query processing: {str(e_outer)}\n{traceback.format_exc()}"
        final_text_on_outer_error = "".join(accumulated_text_obj) 
        final_tool_log_on_outer_error = "".join(accumulated_tool_log_obj)
        print(f"[PROCESS_QUERY_OUTER_ERROR] Final text on outer error: {final_text_on_outer_error!r}")
        print(f"[PROCESS_QUERY_OUTER_ERROR] Final tool log on outer error: {final_tool_log_on_outer_error!r}")
        return {"error": error_msg_outer}, final_text_on_outer_error, final_tool_log_on_outer_error


#new message
async def initialize_session(mcp_config=None):
    with st.spinner("🔄 Connecting to MCP server..."):
        await cleanup_mcp_client()
        if mcp_config is None:
            mcp_config = load_config_from_json()

        client = MultiServerMCPClient(mcp_config)

        try:
            # 🔥 New API usage: call `get_tools()` directly (no `__aenter__`)
            tools = await client.get_tools()
            st.session_state.tool_count = len(tools)
            st.session_state.mcp_client = client
        except Exception as e:
            st.error(f"Failed to connect MCP client or get tools: {e}")
            return False

        selected_model_key = st.session_state.selected_model
        model_token_config = OUTPUT_TOKEN_INFO.get(selected_model_key, {"max_tokens": 4096})
        llm_max_tokens = model_token_config["max_tokens"]

        if selected_model_key in ["claude-3-7-sonnet-latest", "claude-3-5-sonnet-latest", "claude-3-5-haiku-latest"]:
            model = ChatAnthropic(model=selected_model_key, temperature=0.1, max_tokens=llm_max_tokens)
        else:
            model = ChatOpenAI(model=selected_model_key, temperature=0.1, max_tokens=llm_max_tokens)

        try:
            checkpointer = MemorySaver()
            st.session_state.agent = create_react_agent(model, tools, checkpointer=checkpointer, prompt=SYSTEM_PROMPT)
            st.session_state.session_initialized = True
            return True
        except Exception as e:
            st.error(f"Failed to create ReAct agent: {e}")
            import traceback
            st.error(traceback.format_exc())
            return False

            
# --- Sidebar UI ---
with st.sidebar: 
    st.subheader("⚙️ System Settings")
    available_models = []
    has_anthropic_key = os.environ.get("ANTHROPIC_API_KEY") is not None
    if has_anthropic_key:
        available_models.extend([m for m in OUTPUT_TOKEN_INFO if m.startswith("claude-") and m in ["claude-3-7-sonnet-latest", "claude-3-5-sonnet-latest", "claude-3-5-haiku-latest"]])
    has_openai_key = os.environ.get("OPENAI_API_KEY") is not None
    if has_openai_key:
        available_models.extend([m for m in OUTPUT_TOKEN_INFO if m.startswith("gpt-") and m in ["gpt-4o", "gpt-4o-mini"]])
    if not available_models:
        st.warning("⚠️ API keys are not configured. Please add ANTHROPIC_API_KEY or OPENAI_API_KEY to your .env file.")
        available_models = ["claude-3-7-sonnet-latest"] 
    current_model_idx = 0
    if st.session_state.selected_model in available_models:
        current_model_idx = available_models.index(st.session_state.selected_model)
    elif available_models : 
        st.session_state.selected_model = available_models[0]
    previous_model = st.session_state.selected_model
    st.session_state.selected_model = st.selectbox(
        "🤖 Select model to use", options=available_models, index=current_model_idx,
        help="Anthropic models require ANTHROPIC_API_KEY and OpenAI models require OPENAI_API_KEY to be set as environment variables."
    )
    if previous_model != st.session_state.selected_model and st.session_state.session_initialized:
        st.warning("⚠️ Model has been changed. Click 'Apply Settings' button to apply changes.")
    st.session_state.timeout_seconds = st.slider(
        "⏱️ Response generation time limit (seconds)", 60, 300, st.session_state.timeout_seconds, 10,
        help="Set the maximum time for the agent to generate a response. Complex tasks may require more time."
    )
    st.session_state.recursion_limit = st.slider(
        "⏱️ Recursion call limit (count)", 10, 200, st.session_state.recursion_limit, 10,
        help="Set the recursion call limit. Setting too high a value may cause memory issues."
    )
    st.divider()
    st.subheader("🔧 Tool Settings")
    if "mcp_tools_expander" not in st.session_state: st.session_state.mcp_tools_expander = False
    with st.expander("🧰 Add MCP Tools", expanded=st.session_state.mcp_tools_expander):
        loaded_config = load_config_from_json()
        if "pending_mcp_config" not in st.session_state:
            st.session_state.pending_mcp_config = loaded_config.copy()
        example_json = {"github": {"command": "npx","args": ["-y","@smithery/cli@latest","run","@smithery-ai/github","--config",'{"githubPersonalAccessToken":"your_token_here"}'],"transport": "stdio"}}
        default_text = json.dumps(example_json, indent=2, ensure_ascii=False)
        new_tool_json = st.text_area("Tool JSON", default_text, height=250)
        if st.button("Add Tool", type="primary", key="add_tool_button", use_container_width=True):
            try: 
                if not new_tool_json.strip().startswith("{") or not new_tool_json.strip().endswith("}"):
                    st.error("JSON must start and end with curly braces ({}).")
                else:
                    parsed_tool = json.loads(new_tool_json)
                    if "mcpServers" in parsed_tool: parsed_tool = parsed_tool["mcpServers"]
                    if not parsed_tool: st.error("Please enter at least one tool.")
                    else:
                        success_tools = []
                        for tool_name, tool_config in parsed_tool.items():
                            if ("command" not in tool_config and "url" not in tool_config) or \
                               ("command" in tool_config and "args" not in tool_config):
                                st.error(f"'{tool_name}' tool configuration is invalid.")
                            else:
                                st.session_state.pending_mcp_config[tool_name] = tool_config
                                success_tools.append(tool_name)
                        if success_tools:
                            st.success(f"{', '.join(success_tools)} tool(s) have been added. Click 'Apply Settings' button to apply.")
                            st.session_state.mcp_tools_expander = False; st.rerun()
            except Exception as e: st.error(f"Error adding tool: {e}")
    with st.expander("📋 Registered Tools List", expanded=True):
        pending_config_disp = st.session_state.get("pending_mcp_config", {})
        for tool_name in list(pending_config_disp.keys()):
            col1, col2 = st.columns([8, 2])
            col1.markdown(f"- **{tool_name}**")
            if col2.button("Delete", key=f"delete_{tool_name}"):
                del st.session_state.pending_mcp_config[tool_name]
                st.success(f"{tool_name} tool has been deleted. Click 'Apply Settings' button to apply."); st.rerun()
    st.divider()
    st.subheader("📊 System Information")
    st.write(f"🛠️ MCP Tools Count: {st.session_state.get('tool_count', 'Initializing...')}")
    st.write(f"🧠 Current Model: {st.session_state.selected_model}")
    if st.button("Apply Settings", key="apply_button", type="primary", use_container_width=True):
        apply_status = st.empty()
        with apply_status.container():
            #print("starting")
            st.warning("🔄 Applying changes. Please wait...")
            progress_bar = st.progress(0)
            save_config_to_json(st.session_state.pending_mcp_config)
            #print(st.session_state)
            progress_bar.progress(15)
            #print("processing")
            st.session_state.session_initialized = False; st.session_state.agent = None
            progress_bar.progress(30)
            success = st.session_state.event_loop.run_until_complete(
                initialize_session(st.session_state.pending_mcp_config)
            )
            progress_bar.progress(100)
            #print("completed")
            if success: st.success("✅ New settings have been applied.")
            else: st.error("❌ Failed to apply settings.")
        st.rerun()
    
    st.divider()
    st.subheader("🔄 Actions")
    if st.button("Clear Chat History", use_container_width=True, type="primary"): 
        st.session_state.thread_id = random_uuid() 
        st.session_state.history = []
        st.success("✅ Chat history cleared and conversation session reset.")
        st.rerun()

    if use_login and st.session_state.authenticated:
        st.divider()
        if st.button("Logout", use_container_width=True, type="secondary"):
            st.session_state.authenticated = False; st.success("✅ You have been logged out."); st.rerun()

# --- Initialize default session ---
if not st.session_state.session_initialized:
    st.info("MCP server and agent are not initialized. Please click 'Apply Settings' button.")

print_message()

user_query = st.chat_input("💬 Enter your question")
if user_query:
    if st.session_state.session_initialized and st.session_state.agent:
        st.chat_message("user", avatar="🧑‍💻").markdown(user_query)
        with st.chat_message("assistant", avatar="🤖"):
            tool_placeholder = st.empty()
            text_placeholder = st.empty()
            response_obj, final_text, final_tool_log = st.session_state.event_loop.run_until_complete(
                process_query(user_query, text_placeholder, tool_placeholder, st.session_state.timeout_seconds)
            )
        if isinstance(response_obj, dict) and "error" in response_obj: 
            st.error(response_obj["error"])
        else:
            st.session_state.history.append({"role": "user", "content": user_query})
            st.session_state.history.append({"role": "assistant", "content": final_text})
            if final_tool_log and final_tool_log.strip():
                st.session_state.history.append({"role": "assistant_tool", "content": final_tool_log})
        st.rerun()
    else:
        st.warning("⚠️ MCP server and agent are not initialized. Please click 'Apply Settings'.")