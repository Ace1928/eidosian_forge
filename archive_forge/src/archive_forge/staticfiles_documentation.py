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

        Given the request and response headers, return `True` if an HTTP
        "Not Modified" response could be returned instead.
        