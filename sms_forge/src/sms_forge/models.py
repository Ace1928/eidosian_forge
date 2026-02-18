from __future__ import annotations
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class Message(BaseModel):
    """Standardized message model."""
    sender: str
    body: str
    timestamp: datetime = Field(default_factory=datetime.now)
    received: bool = True
    metadata: Dict[str, Any] = {}

class Contact(BaseModel):
    """Eidosian Contact."""
    name: str
    number: str
    last_contact: Optional[datetime] = None
    metadata: Dict[str, Any] = {}
