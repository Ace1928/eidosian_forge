from typing import Any, Iterator, Optional, Sequence
from ..utils.base64 import base64, unbase64
from .connection import (
def cursor_to_offset(cursor: ConnectionCursor) -> Optional[int]:
    """Extract the offset from the cursor string."""
    try:
        return int(unbase64(cursor)[len(PREFIX):])
    except ValueError:
        return None