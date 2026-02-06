#!/usr/bin/env python3
"""
Standardized API Schemas and Response Envelopes for Moltbook Nexus.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Generic, List, Optional, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")

class NexusMetadata(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = Field(default_factory=time.time)
    processing_time_ms: Optional[float] = None
    version: str = "2.0.0"

class NexusResponse(BaseModel, Generic[T]):
    """Unified response envelope for all Nexus API calls."""
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    meta: NexusMetadata = Field(default_factory=NexusMetadata)

    @classmethod
    def ok(cls, data: T, start_time: float) -> NexusResponse[T]:
        meta = NexusMetadata(processing_time_ms=(time.time() - start_time) * 1000)
        return cls(success=True, data=data, meta=meta)

    @classmethod
    def fail(cls, error: str, start_time: float) -> NexusResponse[T]:
        meta = NexusMetadata(processing_time_ms=(time.time() - start_time) * 1000)
        return cls(success=False, error=error, meta=meta)

class ReputationUpdate(BaseModel):
    username: str
    delta: float
    new_score: float
    reason: Optional[str] = None
