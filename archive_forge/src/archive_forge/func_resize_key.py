from __future__ import annotations
import uuid
import hashlib
import base64
from lazyops.libs.pooler import ThreadPooler
from lazyops.imports._pycryptodome import resolve_pycryptodome
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
from typing import Any, Optional, List, Dict
def resize_key(key: str, length: int=16) -> str:
    """
    Resizes the Key
    """
    return key.rjust((len(key) // length + 1) * length)