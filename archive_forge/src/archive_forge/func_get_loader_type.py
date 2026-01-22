from __future__ import annotations
import logging
import os
import pathlib
import platform
from typing import Optional, Tuple
from langchain_core.env import get_runtime_environment
from langchain_core.pydantic_v1 import BaseModel
from langchain_community.document_loaders.base import BaseLoader
def get_loader_type(loader: str) -> str:
    """Return loader type among, file, dir or in-memory.

    Args:
        loader (str): Name of the loader, whose type is to be resolved.

    Returns:
        str: One of the loader type among, file/dir/in-memory.
    """
    for loader_type, loaders in LOADER_TYPE_MAPPING.items():
        if loader in loaders:
            return loader_type
    return 'unsupported'