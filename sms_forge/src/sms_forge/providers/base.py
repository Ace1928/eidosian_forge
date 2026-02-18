from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class SmsMessage(BaseModel):
    """Unified SMS message format."""
    id: Optional[str] = None
    sender: str
    body: str
    timestamp: datetime = Field(default_factory=datetime.now)
    received: bool = True

class SmsProvider(ABC):
    """Abstract Base Class for SMS backends."""
    
    @abstractmethod
    async def send(self, recipient: str, message: str) -> bool:
        """Send an SMS message."""
        pass

    @abstractmethod
    async def list_messages(self, limit: int = 10) -> List[SmsMessage]:
        """Retrieve recent SMS messages."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is functional in the current environment."""
        pass
