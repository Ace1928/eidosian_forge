from typing import Union, Iterable, Sequence, Any, Optional, Iterator
import sys
import json as _builtin_json
import gzip
from . import ujson
from .util import force_path, force_string, FilePath, JSONInput, JSONOutput
def is_json_serializable(obj: Any) -> bool:
    """Check if a Python object is JSON-serializable.

    obj: The object to check.
    RETURNS (bool): Whether the object is JSON-serializable.
    """
    if hasattr(obj, '__call__'):
        return False
    try:
        ujson.dumps(obj)
        return True
    except (TypeError, OverflowError):
        return False