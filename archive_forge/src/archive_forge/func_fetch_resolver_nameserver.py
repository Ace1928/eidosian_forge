import os
import time
import socket
import pathlib
import tempfile
import contextlib
from typing import Union, Optional
from functools import lru_cache
@lru_cache()
def fetch_resolver_nameserver(path: Optional[str]='/etc/resolv.conf') -> Optional[str]:
    """
    Fetches the nameserver from the resolv.conf
    """
    if path is None:
        return None
    p = pathlib.Path(path)
    if not p.exists():
        return None
    lines = p.read_text().splitlines()
    return next((line.split(' ')[-1].strip() for line in lines if line.startswith('nameserver')), None)