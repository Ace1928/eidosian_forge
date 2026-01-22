from __future__ import annotations
import os
from typing import Any, Dict, Optional
def get_from_dict_or_env(data: Dict[str, Any], key: str, env_key: str, default: Optional[str]=None) -> str:
    """Get a value from a dictionary or an environment variable."""
    if key in data and data[key]:
        return data[key]
    else:
        return get_from_env(key, env_key, default=default)