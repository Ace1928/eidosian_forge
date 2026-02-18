from __future__ import annotations
import json
import subprocess
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from .models import Message
from eidosian_core import eidosian

class UnifiedSignal(BaseModel):
    """A message signal from any source (SMS, Notification, Content Provider)."""
    source: str # 'sms', 'notification', 'mms'
    timestamp: datetime
    sender: str
    body: str
    metadata: Dict[str, Any] = {}

class AdvancedRetrieval:
    """The advanced communication sensor for Eidos."""

    @eidosian()
    async def get_unified_signals(self, limit: int = 10) -> List[UnifiedSignal]:
        """Aggregate signals from multiple sources."""
        signals = []
        
        # 1. Capture Notifications (RCS/Signal/WhatsApp/MMS capture)
        signals.extend(await self._get_notification_signals())
        
        # 2. Capture Legacy SMS
        signals.extend(await self._get_sms_signals(limit))
        
        # Sort by timestamp descending
        signals.sort(key=lambda x: x.timestamp, reverse=True)
        return signals[:limit]

    async def _get_notification_signals(self) -> List[UnifiedSignal]:
        try:
            raw = subprocess.check_output(["termux-notification-list"]).decode()
            data = json.loads(raw)
            signals = []
            # Known messaging package names
            msg_packages = [
                "com.google.android.apps.messaging", # Google Messages (RCS/SMS)
                "org.thoughtcrime.securesms",        # Signal
                "com.whatsapp",                      # WhatsApp
                "com.facebook.orca"                  # Messenger
            ]
            
            for n in data:
                if n.get("packageName") in msg_packages:
                    signals.append(UnifiedSignal(
                        source=f"notification:{n['packageName']}",
                        timestamp=datetime.now(), # Notifications don't always have accurate timestamps in the list
                        sender=n.get("title", "Unknown"),
                        body=n.get("content", ""),
                        metadata={"id": n.get("id"), "tag": n.get("tag")}
                    ))
            return signals
        except Exception:
            return []

    async def _get_sms_signals(self, limit: int) -> List[UnifiedSignal]:
        try:
            # Note: We reuse the existing logic but wrap it in UnifiedSignal
            # For now, calling termux-sms-list directly for simplicity in this module
            raw = subprocess.check_output(["termux-sms-list", "-l", str(limit)]).decode()
            data = json.loads(raw)
            signals = []
            for m in data:
                signals.append(UnifiedSignal(
                    source="sms",
                    timestamp=datetime.strptime(m["received"], "%Y-%m-%d %H:%M:%S") if "received" in m else datetime.now(),
                    sender=m.get("number", "Unknown"),
                    body=m.get("body", ""),
                    metadata={"type": m.get("type"), "read": m.get("read")}
                ))
            return signals
        except Exception:
            return []
