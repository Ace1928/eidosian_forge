from __future__ import annotations
import io
import json
import os
import typing as t
from .encoding import (
def read_text_file(path: str) -> str:
    """Return the contents of the specified path as text."""
    return to_text(read_binary_file(path))