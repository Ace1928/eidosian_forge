from enum import Enum
from typing import (
from fastapi import routing
from fastapi.datastructures import Default, DefaultPlaceholder
from fastapi.exception_handlers import (
from fastapi.exceptions import RequestValidationError, WebSocketRequestValidationError
from fastapi.logger import logger
from fastapi.openapi.docs import (
from fastapi.openapi.utils import get_openapi
from fastapi.params import Depends
from fastapi.types import DecoratedCallable, IncEx
from fastapi.utils import generate_unique_id
from starlette.applications import Starlette
from starlette.datastructures import State
from starlette.exceptions import HTTPException
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.routing import BaseRoute
from starlette.types import ASGIApp, Lifespan, Receive, Scope, Send
from typing_extensions import Annotated, Doc, deprecated  # type: ignore [attr-defined]
def middleware(self, middleware_type: Annotated[str, Doc('\n                The type of middleware. Currently only supports `http`.\n                ')]) -> Callable[[DecoratedCallable], DecoratedCallable]:
    """
        Add a middleware to the application.

        Read more about it in the
        [FastAPI docs for Middleware](https://fastapi.tiangolo.com/tutorial/middleware/).

        ## Example

        ```python
        import time

        from fastapi import FastAPI, Request

        app = FastAPI()


        @app.middleware("http")
        async def add_process_time_header(request: Request, call_next):
            start_time = time.time()
            response = await call_next(request)
            process_time = time.time() - start_time
            response.headers["X-Process-Time"] = str(process_time)
            return response
        ```
        """

    def decorator(func: DecoratedCallable) -> DecoratedCallable:
        self.add_middleware(BaseHTTPMiddleware, dispatch=func)
        return func
    return decorator