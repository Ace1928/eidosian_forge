import os
from typing import List, Optional
from .providers.base import SmsProvider, SmsMessage
from .providers.termux import TermuxProvider
from .providers.twilio import TwilioProvider
from .utils.parser import extract_code
from eidosian_core import eidosian

from .utils.contacts import ContactManager
from memory_forge import MemoryForge

class SmsForge:
    """The high-level SMS interface for Eidosian agents."""

    def __init__(self, provider: Optional[SmsProvider] = None):
        self.contacts = ContactManager()
        self.memory = MemoryForge()
        if provider:
            self.provider = provider
        else:
            # Auto-detection logic
            termux = TermuxProvider()
            if termux.is_available():
                self.provider = termux
            else:
                self.provider = TwilioProvider()

    @eidosian
    async def send_message(self, recipient: str, body: str) -> bool:
        """Send an SMS and log to memory."""
        # Resolve name if possible
        target = self.contacts.get_number(recipient) or recipient
        success = await self.provider.send(target, body)
        
        if success:
            self.memory.remember(
                content=f"SENT SMS to {recipient}: {body}",
                metadata={"type": "sms", "direction": "sent", "recipient": recipient}
            )
        return success

    @eidosian
    async def get_latest_messages(self, limit: int = 10, address: Optional[str] = None) -> List[SmsMessage]:
        """Fetch recent messages and log discovery to memory."""
        msgs = await self.provider.list_messages(limit, address=address)
        return msgs

    @eidosian
    async def get_messages_by_contact(self, name: str, limit: int = 10) -> List[SmsMessage]:
        """Fetch messages for a specific named contact."""
        number = self.contacts.get_number(name)
        if not number:
            return []
        return await self.get_latest_messages(limit=limit, address=number)

    @eidosian
    async def get_2fa_code(self, sender_pattern: Optional[str] = None) -> Optional[str]:
        """
        Poll for the latest 2FA code. 
        If sender_pattern is provided, filters for that sender.
        """
        messages = await self.get_latest_messages(limit=5)
        for msg in messages:
            if sender_pattern and sender_pattern.lower() not in msg.sender.lower():
                continue
            code = extract_code(msg.body)
            if code:
                return code
        return None
