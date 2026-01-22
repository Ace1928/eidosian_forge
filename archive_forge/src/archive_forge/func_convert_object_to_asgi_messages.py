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
def convert_object_to_asgi_messages(obj: Optional[Any]=None, status_code: int=200) -> List[Message]:
    """Serializes the provided object and converts it to ASGI messages.

    These ASGI messages can be sent via an ASGI `send` interface to comprise an HTTP
    response.
    """
    body = None
    content_type = None
    if obj is None:
        body = b''
        content_type = b'text/plain'
    elif isinstance(obj, bytes):
        body = obj
        content_type = b'text/plain'
    elif isinstance(obj, str):
        body = obj.encode('utf-8')
        content_type = b'text/plain; charset=utf-8'
    else:
        body = json.dumps(jsonable_encoder(obj, custom_encoder=serve_encoders), separators=(',', ':')).encode()
        content_type = b'application/json'
    return [{'type': 'http.response.start', 'status': status_code, 'headers': [[b'content-type', content_type]]}, {'type': 'http.response.body', 'body': body}]