from __future__ import annotations
import io
import os
import typing
from pathlib import Path
from ._types import (
from ._utils import (
def get_multipart_boundary_from_content_type(content_type: bytes | None) -> bytes | None:
    if not content_type or not content_type.startswith(b'multipart/form-data'):
        return None
    if b';' in content_type:
        for section in content_type.split(b';'):
            if section.strip().lower().startswith(b'boundary='):
                return section.strip()[len(b'boundary='):].strip(b'"')
    return None