from __future__ import annotations
import importlib.util
import os
import stat
import typing
from email.utils import parsedate
import anyio
import anyio.to_thread
from starlette._utils import get_route_path
from starlette.datastructures import URL, Headers
from starlette.exceptions import HTTPException
from starlette.responses import FileResponse, RedirectResponse, Response
from starlette.types import Receive, Scope, Send
def is_not_modified(self, response_headers: Headers, request_headers: Headers) -> bool:
    """
        Given the request and response headers, return `True` if an HTTP
        "Not Modified" response could be returned instead.
        """
    try:
        if_none_match = request_headers['if-none-match']
        etag = response_headers['etag']
        if etag in [tag.strip(' W/') for tag in if_none_match.split(',')]:
            return True
    except KeyError:
        pass
    try:
        if_modified_since = parsedate(request_headers['if-modified-since'])
        last_modified = parsedate(response_headers['last-modified'])
        if if_modified_since is not None and last_modified is not None and (if_modified_since >= last_modified):
            return True
    except KeyError:
        pass
    return False