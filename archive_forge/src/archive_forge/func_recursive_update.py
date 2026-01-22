from __future__ import annotations
import errno
import json
import os
from typing import Any
from traitlets.config import LoggingConfigurable
from traitlets.traitlets import Unicode
def recursive_update(target: dict[Any, Any], new: dict[Any, Any]) -> None:
    """Recursively update one dictionary using another.

    None values will delete their keys.
    """
    for k, v in new.items():
        if isinstance(v, dict):
            if k not in target:
                target[k] = {}
            recursive_update(target[k], v)
            if not target[k]:
                del target[k]
        elif v is None:
            target.pop(k, None)
        else:
            target[k] = v