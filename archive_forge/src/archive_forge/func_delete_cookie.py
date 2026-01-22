from __future__ import annotations
import http.cookies
import json
import os
import stat
import typing
import warnings
from datetime import datetime
from email.utils import format_datetime, formatdate
from functools import partial
from mimetypes import guess_type
from urllib.parse import quote
import anyio
import anyio.to_thread
from starlette._compat import md5_hexdigest
from starlette.background import BackgroundTask
from starlette.concurrency import iterate_in_threadpool
from starlette.datastructures import URL, MutableHeaders
from starlette.types import Receive, Scope, Send
def delete_cookie(self, key: str, path: str='/', domain: str | None=None, secure: bool=False, httponly: bool=False, samesite: typing.Literal['lax', 'strict', 'none'] | None='lax') -> None:
    self.set_cookie(key, max_age=0, expires=0, path=path, domain=domain, secure=secure, httponly=httponly, samesite=samesite)