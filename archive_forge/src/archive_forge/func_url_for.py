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
def url_for(self, name: str, /, **path_params: typing.Any) -> URL:
    router: Router = self.scope['router']
    url_path = router.url_path_for(name, **path_params)
    return url_path.make_absolute_url(base_url=self.base_url)