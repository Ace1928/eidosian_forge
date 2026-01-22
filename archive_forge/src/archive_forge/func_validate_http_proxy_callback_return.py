import asyncio
import inspect
import json
import logging
import pickle
import socket
from typing import Any, List, Optional, Type
import starlette
from fastapi.encoders import jsonable_encoder
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from uvicorn.config import Config
from uvicorn.lifespan.on import LifespanOn
from ray._private.pydantic_compat import IS_PYDANTIC_2
from ray.actor import ActorHandle
from ray.serve._private.constants import SERVE_LOGGER_NAME
from ray.serve._private.utils import serve_encoders
from ray.serve.exceptions import RayServeException
def validate_http_proxy_callback_return(middlewares: Any) -> [starlette.middleware.Middleware]:
    """Validate the return value of HTTP proxy callback.

    Middlewares should be a list of Starlette middlewares. If it is None, we
    will treat it as an empty list. If it is not a list, we will raise an
    error. If it is a list, we will check if all the items in the list are
    Starlette middlewares.
    """
    if middlewares is None:
        middlewares = []
    if not isinstance(middlewares, list):
        raise ValueError('HTTP proxy callback must return a list of Starlette middlewares.')
    else:
        for middleware in middlewares:
            if not issubclass(type(middleware), starlette.middleware.Middleware):
                raise ValueError(f'HTTP proxy callback must return a list of Starlette middlewares, instead got {type(middleware)} type item in the list.')
    return middlewares