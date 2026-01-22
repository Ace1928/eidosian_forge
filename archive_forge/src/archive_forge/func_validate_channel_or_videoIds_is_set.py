from __future__ import annotations
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
from urllib.parse import parse_qs, urlparse
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import root_validator
from langchain_core.pydantic_v1.dataclasses import dataclass
from langchain_community.document_loaders.base import BaseLoader
@root_validator
def validate_channel_or_videoIds_is_set(cls, values: Dict[str, Any]) -> Dict[str, Any]:
    """Validate that either folder_id or document_ids is set, but not both."""
    if not values.get('channel_name') and (not values.get('video_ids')):
        raise ValueError('Must specify either channel_name or video_ids')
    return values