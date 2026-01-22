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
@classmethod
def from_youtube_url(cls, youtube_url: str, **kwargs: Any) -> YoutubeLoader:
    """Given youtube URL, load video."""
    video_id = cls.extract_video_id(youtube_url)
    return cls(video_id, **kwargs)