import json
import subprocess
import asyncio
from typing import List
from datetime import datetime
from .base import SmsProvider, SmsMessage
from eidosian_core import eidosian

class TermuxProvider(SmsProvider):
    """Local hardware provider using Termux-API."""

    def is_available(self) -> bool:
        try:
            subprocess.run(["termux-sms-list", "-h"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    @eidosian
    async def send(self, recipient: str, message: str) -> bool:
        """Send SMS using termux-sms-send."""
        cmd = ["termux-sms-send", "-n", recipient, message]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await proc.communicate()
        return proc.returncode == 0

    @eidosian
    async def list_messages(self, limit: int = 10) -> List[SmsMessage]:
        """List messages using termux-sms-list."""
        cmd = ["termux-sms-list", "-l", str(limit)]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        
        if proc.returncode != 0:
            return []

        try:
            raw_data = json.loads(stdout.decode())
            messages = []
            for item in raw_data:
                # Termux returns dates like "2026-02-18 07:15:00"
                ts = datetime.strptime(item.get("received"), "%Y-%m-%d %H:%M:%S")
                messages.append(SmsMessage(
                    id=str(item.get("_id")),
                    sender=item.get("number"),
                    body=item.get("body"),
                    timestamp=ts,
                    received=True
                ))
            return messages
        except (json.JSONDecodeError, ValueError):
            return []
