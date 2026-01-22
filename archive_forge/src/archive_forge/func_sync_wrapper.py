from __future__ import annotations
import functools
import inspect
import sys
import typing
from urllib.parse import urlencode
from starlette._utils import is_async_callable
from starlette.exceptions import HTTPException
from starlette.requests import HTTPConnection, Request
from starlette.responses import RedirectResponse
from starlette.websockets import WebSocket
@functools.wraps(func)
def sync_wrapper(*args: _P.args, **kwargs: _P.kwargs) -> typing.Any:
    request = kwargs.get('request', args[idx] if idx < len(args) else None)
    assert isinstance(request, Request)
    if not has_required_scope(request, scopes_list):
        if redirect is not None:
            orig_request_qparam = urlencode({'next': str(request.url)})
            next_url = '{redirect_path}?{orig_request}'.format(redirect_path=request.url_for(redirect), orig_request=orig_request_qparam)
            return RedirectResponse(url=next_url, status_code=303)
        raise HTTPException(status_code=status_code)
    return func(*args, **kwargs)