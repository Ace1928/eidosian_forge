from __future__ import annotations
import typing
from shlex import shlex
from urllib.parse import SplitResult, parse_qsl, urlencode, urlsplit
from starlette.concurrency import run_in_threadpool
from starlette.types import Scope
def make_absolute_url(self, base_url: str | URL) -> URL:
    if isinstance(base_url, str):
        base_url = URL(base_url)
    if self.protocol:
        scheme = {'http': {True: 'https', False: 'http'}, 'websocket': {True: 'wss', False: 'ws'}}[self.protocol][base_url.is_secure]
    else:
        scheme = base_url.scheme
    netloc = self.host or base_url.netloc
    path = base_url.path.rstrip('/') + str(self)
    return URL(scheme=scheme, netloc=netloc, path=path)