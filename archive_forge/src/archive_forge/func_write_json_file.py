from __future__ import annotations
import io
import json
import os
import typing as t
from .encoding import (
def write_json_file(path: str, content: t.Any, create_directories: bool=False, formatted: bool=True, encoder: t.Optional[t.Type[json.JSONEncoder]]=None) -> str:
    """Write the given json content to the specified path, optionally creating missing directories."""
    text_content = json.dumps(content, sort_keys=formatted, indent=4 if formatted else None, separators=(', ', ': ') if formatted else (',', ':'), cls=encoder) + '\n'
    write_text_file(path, text_content, create_directories=create_directories)
    return text_content