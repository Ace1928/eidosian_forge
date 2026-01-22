from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, List, Optional, Union
from langchain_core.chat_sessions import ChatSession
from langchain_core.messages import HumanMessage
from langchain_community.chat_loaders.base import BaseChatLoader
def nanoseconds_from_2001_to_datetime(nanoseconds: int) -> datetime:
    timestamp_in_seconds = nanoseconds / 1000000000.0
    reference_date_seconds = datetime(2001, 1, 1).timestamp()
    actual_timestamp = reference_date_seconds + timestamp_in_seconds
    return datetime.fromtimestamp(actual_timestamp)