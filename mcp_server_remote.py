# zwigato_mcp_server.py

from mcp.server.fastmcp import FastMCP

# To Do: --- Import Zwigato Mock Data Sources ---
"""
from $$ import (
    ,
    ,

)
"""

# --- MCP Server Setup ---
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8007

mcp = FastMCP(
    "ZwigatoSupportAssistantTools",
    instructions=(
        // To Do: --- Instructions for the MCP server ---
    ),
    host=SERVER_HOST,
    port=SERVER_PORT,
)

# --- MCP Tools ---

@mcp.tool()
async def search_wiki(query: str) -> str:
    """
    Searches the Zwigato WIKI for information based on a query.
    Use this to answer questions about Zwigato services, delivery fees, membership, policies, etc.

    Args:
        query (str): The search term or question from the customer.
                     Example: "Zwigato Gold benefits", "refund policy", "delivery fees"

    Returns:
        str: Information found in the WIKI related to the query, or a message if nothing is found.
    """
    print(f"[MCP WIKI Server] Received search_wiki request with query: '{query}'")
    query_lower = query.lower()
    # Prioritize exact or near-exact keyword matches
    for keyword, content in _MOCK_WIKI_DB.items():
        if keyword in query_lower or query_lower in keyword:
            print(f"[MCP WIKI Server] Found direct match for '{keyword}'.")
            return content
    # Fallback to content search
    for keyword, content in _MOCK_WIKI_DB.items():
        if query_lower in content.lower():
            print(f"[MCP WIKI Server] Found partial content match for query '{query}' in content for '{keyword}'.")
            return content
    print(f"[MCP WIKI Server] No information found for query: '{query}'.")
    return f"I couldn't find specific information for '{query}' in our Zwigato WIKI. Could you rephrase or ask about something else?"

@mcp.tool()
async def read_order_status(order_id: str) -> str:
    """
    Reads the current status and details of a customer's Zwigato order using the order ID.

    Args:
        order_id (str): The unique identifier for the order. Example: "ORDZW001"

    Returns:
        str: A message indicating the order details and status, or an error if the order is not found.
    """
    print(f"[MCP Order Server] Received read_order_status request for order_id: '{order_id}'")
    order = _MOCK_ORDERS_DB.get(order_id)
    if order:
        restaurant_name = "Unknown Restaurant"
        if order.get("restaurant_id") and order["restaurant_id"] in _MOCK_RESTAURANTS_DB:
            restaurant_name = _MOCK_RESTAURANTS_DB[order["restaurant_id"]]["name"]

        items_str = ", ".join(order.get("items", ["No items listed"]))
        status_message = (
            f"Order '{order_id}': Status is '{order['status']}'. "
            f"Items: {items_str}. "
            f"Restaurant: {restaurant_name}. "
            f"Estimated Delivery: {order.get('estimated_delivery_time', 'N/A')}. "
        )
        if order.get("special_instructions"):
            status_message += f"Special Instructions: {order['special_instructions']}"

        print(f"[MCP Order Server] {status_message}")
        return status_message
    else:
        error_message = f"Sorry, I could not find a Zwigato order with ID '{order_id}'."
        print(f"[MCP Order Server] {error_message}")
        return error_message

@mcp.tool()
async def update_order_status(order_id: str, new_status: str) -> str:
    """
    Updates the status of an existing Zwigato customer order, primarily for cancellation.
    For cancellations, set new_status to 'cancelled'.

    Args:
        order_id (str): The unique identifier for the order to be updated. Example: "ORDZW001"
        new_status (str): The new status to set. For cancellation, this must be 'cancelled'.

    Returns:
        str: A confirmation message if the update was successful, or an error/reason if not.
    """
    print(f"[MCP Order Server] Received update_order_status request for order_id: '{order_id}', new_status: '{new_status}'")
    order = _MOCK_ORDERS_DB.get(order_id) # _MOCK_ORDERS_DB is now imported

    if not order:
        error_message = f"Sorry, I could not find a Zwigato order with ID '{order_id}' to update."
        print(f"[MCP Order Server] {error_message}")
        return error_message

    normalized_new_status = new_status.lower()

    if normalized_new_status == "cancelled":
        current_status_lower = order["status"].lower()
        cancellable_statuses = ["order placed", "awaiting rider assignment"]

        if current_status_lower in cancellable_statuses:
            order["status"] = "Cancelled"
            order["estimated_delivery_time"] = None
            # _MOCK_ORDERS_DB[order_id] = order # This line might not be strictly necessary if 'order' is a reference
                                              # to the dictionary item, but it's safer to be explicit for mutable operations
                                              # on shared data structures like dictionaries.
            message = f"Order '{order_id}' has been successfully cancelled."
            print(f"[MCP Order Server] {message}")
            return message
        elif current_status_lower == "cancelled":
            message = f"Order '{order_id}' is already cancelled."
            print(f"[MCP Order Server] {message}")
            return message
        else:
            message = f"Order '{order_id}' cannot be cancelled because its current status is '{order['status']}'. Please refer to our cancellation policy or contact support for more help."
            print(f"[MCP Order Server] {message}")
            return message
    else:
        message = f"This tool is primarily for cancelling orders. To change status to '{new_status}' is not supported via this action for order '{order_id}'. Current status remains '{order['status']}'."
        print(f"[MCP Order Server] {message}")
        return message


if __name__ == "__main__":
    print("MCP server for Zwigato Support Assistant Tools is running...")
    print(f"Access it at http://{SERVER_HOST}:{SERVER_PORT}")
    # print("Available tools via MCP:")
    # for tool_name in mcp.tools.keys():
    #     print(f"  - {tool_name}")
    # print("\n--- Mock Data (Imported - Sample) ---")
    # print("First Order (ORDZW001):", _MOCK_ORDERS_DB.get("ORDZW001"))
    # print("Wiki Entry (refund policy):", _MOCK_WIKI_DB.get("refund policy"))
    # print("-----------------------------\n")

    mcp.run(transport="stdio")