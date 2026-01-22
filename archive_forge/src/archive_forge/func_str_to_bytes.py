from __future__ import annotations
import sys
import traceback
def str_to_bytes(s):
    """Convert str to bytes."""
    if isinstance(s, str):
        return s.encode()
    return s