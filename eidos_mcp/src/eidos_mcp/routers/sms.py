from __future__ import annotations

from typing import Optional

from ..forge_loader import ensure_forge_import

# Ensure dependencies are in path
ensure_forge_import("memory_forge")
ensure_forge_import("sms_forge")

from eidosian_core import eidosian
from sms_forge.core import SmsForge

from ..core import tool

# Initialize the forge
sms = SmsForge()


@tool(
    name="sms_send",
    description="Send an SMS message using a number or contact name.",
    parameters={
        "type": "object",
        "properties": {
            "recipient": {"type": "string", "description": "Phone number or Contact Name (if saved)."},
            "body": {"type": "string", "description": "The message content."},
        },
        "required": ["recipient", "body"],
    },
)
@eidosian()
async def sms_send(recipient: str, body: str) -> str:
    """Send an SMS message using a number or contact name."""
    success = await sms.send_message(recipient, body)
    return f"SMS Sent to {recipient} Successfully" if success else f"Failed to send SMS to {recipient}"


@tool(
    name="sms_add_contact",
    description="Add a contact to the Eidosian directory.",
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "number": {"type": "string"},
        },
        "required": ["name", "number"],
    },
)
@eidosian()
async def sms_add_contact(name: str, number: str) -> str:
    """Add a contact to the Eidosian directory."""
    sms.contacts.add_contact(name, number)
    return f"Contact '{name}' saved with number {number}."


@tool(
    name="sms_list",
    description="List the latest SMS messages, optionally filtered by contact name.",
    parameters={
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "default": 5},
            "contact_name": {"type": "string", "description": "Optional name to filter by."},
        },
    },
)
@eidosian()
async def sms_list(limit: int = 5, contact_name: Optional[str] = None) -> str:
    """List the latest SMS messages, optionally filtered by contact name."""
    if contact_name:
        msgs = await sms.get_messages_by_contact(contact_name, limit)
    else:
        msgs = await sms.get_latest_messages(limit)

    if not msgs:
        return "No messages found."

    out = []
    for m in msgs:
        direction = "INBOX" if m.received else "SENT"
        out.append(f"[{m.timestamp}] {direction} | {m.sender}: {m.body}")
    return "\n".join(out)
