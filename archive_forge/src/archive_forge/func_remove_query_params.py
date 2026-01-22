from __future__ import annotations
import typing
from shlex import shlex
from urllib.parse import SplitResult, parse_qsl, urlencode, urlsplit
from starlette.concurrency import run_in_threadpool
from starlette.types import Scope
def remove_query_params(self, keys: str | typing.Sequence[str]) -> 'URL':
    if isinstance(keys, str):
        keys = [keys]
    params = MultiDict(parse_qsl(self.query, keep_blank_values=True))
    for key in keys:
        params.pop(key, None)
    query = urlencode(params.multi_items())
    return self.replace(query=query)