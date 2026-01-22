import html
import json
import re
import urllib.parse
from tornado.util import unicode_type
import typing
from typing import Union, Any, Optional, Dict, List, Callable
def parse_qs_bytes(qs: Union[str, bytes], keep_blank_values: bool=False, strict_parsing: bool=False) -> Dict[str, List[bytes]]:
    """Parses a query string like urlparse.parse_qs,
    but takes bytes and returns the values as byte strings.

    Keys still become type str (interpreted as latin1 in python3!)
    because it's too painful to keep them as byte strings in
    python3 and in practice they're nearly always ascii anyway.
    """
    if isinstance(qs, bytes):
        qs = qs.decode('latin1')
    result = urllib.parse.parse_qs(qs, keep_blank_values, strict_parsing, encoding='latin1', errors='strict')
    encoded = {}
    for k, v in result.items():
        encoded[k] = [i.encode('latin1') for i in v]
    return encoded