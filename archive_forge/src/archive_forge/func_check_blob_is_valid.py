from __future__ import annotations
import contextlib
import mimetypes
from abc import ABC, abstractmethod
from io import BufferedReader, BytesIO
from pathlib import PurePath
from typing import Any, Dict, Generator, Iterable, Mapping, Optional, Union, cast
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
@root_validator(pre=True)
def check_blob_is_valid(cls, values: Mapping[str, Any]) -> Mapping[str, Any]:
    """Verify that either data or path is provided."""
    if 'data' not in values and 'path' not in values:
        raise ValueError('Either data or path must be provided')
    return values