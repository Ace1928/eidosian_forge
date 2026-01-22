from __future__ import annotations
from collections.abc import Mapping
from functools import partial
from typing import NamedTuple
from urllib.parse import parse_qsl, quote, unquote, urlparse
from ..log import get_logger
def url_to_parts(url):
    """Parse URL into :class:`urlparts` tuple of components."""
    scheme = urlparse(url).scheme
    schemeless = url[len(scheme) + 3:]
    parts = urlparse('http://' + schemeless)
    path = parts.path or ''
    path = path[1:] if path and path[0] == '/' else path
    return urlparts(scheme, unquote(parts.hostname or '') or None, parts.port, unquote(parts.username or '') or None, unquote(parts.password or '') or None, unquote(path or '') or None, dict(parse_qsl(parts.query)))