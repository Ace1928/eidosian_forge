from __future__ import annotations
from typing import List, Optional
from sms_forge.core import SmsForge
from ..core import mcp

# Initialize the forge
sms = SmsForge()

@mcp.tool()
async def sms_send(recipient: str, body: str) -> str:
    """
    Send an SMS message using the best available provider (Termux or Twilio).
    :param recipient: Phone number in international format.
    :param body: The message content.
    """
    success = await sms.send_message(recipient, body)
    return "SMS Sent Successfully" if success else "Failed to send SMS"

@mcp.tool()
async def sms_get_codes(sender_pattern: Optional[str] = None) -> str:
    """
    Poll recent SMS for 2FA or verification codes.
    :param sender_pattern: Optional string to filter sender name (e.g. 'Google', 'Bank').
    """
    code = await sms.get_2fa_code(sender_pattern)
    return f"Latest Code: {code}" if code else "No verification code found in recent messages."

@mcp.tool()
async def sms_list(limit: int = 5) -> str:
    """
    List the latest SMS messages.
    """
    msgs = await sms.get_latest_messages(limit)
    if not msgs:
        return "No messages found."
    
    out = []
    for m in msgs:
        out.append(f"[{m.timestamp}] {m.sender}: {m.body}")
    return "
".join(out)
