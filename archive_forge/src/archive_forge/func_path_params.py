from __future__ import annotations
import json
import typing
from http import cookies as http_cookies
import anyio
from starlette._utils import AwaitableOrContextManager, AwaitableOrContextManagerWrapper
from starlette.datastructures import URL, Address, FormData, Headers, QueryParams, State
from starlette.exceptions import HTTPException
from starlette.formparsers import FormParser, MultiPartException, MultiPartParser
from starlette.types import Message, Receive, Scope, Send
@property
def path_params(self) -> dict[str, typing.Any]:
    return self.scope.get('path_params', {})