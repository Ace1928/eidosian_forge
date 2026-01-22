from __future__ import annotations
import logging
import os
import tempfile
from abc import abstractmethod
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Sequence, Union
from langchain_core.pydantic_v1 import (
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.blob_loaders.file_system import (
from langchain_community.document_loaders.blob_loaders.schema import Blob
def fetch_mime_types(file_types: Sequence[_FileType]) -> Dict[str, str]:
    """Fetch the mime types for the specified file types."""
    mime_types_mapping = {}
    for file_type in file_types:
        if file_type.value == 'doc':
            mime_types_mapping[file_type.value] = 'application/msword'
        elif file_type.value == 'docx':
            mime_types_mapping[file_type.value] = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        elif file_type.value == 'pdf':
            mime_types_mapping[file_type.value] = 'application/pdf'
    return mime_types_mapping