import html
import inspect
import traceback
import typing
from starlette._utils import is_async_callable
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import HTMLResponse, PlainTextResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send
def debug_response(self, request: Request, exc: Exception) -> Response:
    accept = request.headers.get('accept', '')
    if 'text/html' in accept:
        content = self.generate_html(exc)
        return HTMLResponse(content, status_code=500)
    content = self.generate_plain_text(exc)
    return PlainTextResponse(content, status_code=500)