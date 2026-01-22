from __future__ import annotations
import uuid
import hashlib
import base64
from lazyops.libs.pooler import ThreadPooler
from lazyops.imports._pycryptodome import resolve_pycryptodome
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
from typing import Any, Optional, List, Dict
def parse_scopes(scope: Optional[str]=None, scopes: Optional[List[str]]=None) -> Optional[List[str]]:
    """
    Parses the Scopes
    """
    if scopes is None:
        scopes = []
    if scope is not None:
        if ' ' in scope:
            scopes.extend(scope.split(' '))
        else:
            scopes.append(scope)
    return scopes or None