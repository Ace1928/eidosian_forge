import os
from typing import List, Optional
from .providers.base import SmsProvider, SmsMessage
from .providers.termux import TermuxProvider
from .providers.twilio import TwilioProvider
from .utils.parser import extract_code
from eidosian_core import eidosian

class SmsForge:
    """The high-level SMS interface for Eidosian agents."""

    def __init__(self, provider: Optional[SmsProvider] = None):
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
        """Send an SMS via the active provider."""
        return await self.provider.send(recipient, body)

    @eidosian
    async def get_latest_messages(self, limit: int = 10) -> List[SmsMessage]:
        """Fetch recent messages."""
        return await self.provider.list_messages(limit)

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
