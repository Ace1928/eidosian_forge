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
def make_buffered_asgi_receive(serialized_body: bytes) -> Receive:
    """Returns an ASGI receiver that returns the provided buffered body."""
    received = False

    async def mock_receive():
        nonlocal received
        if received:
            block_forever = asyncio.Event()
            await block_forever.wait()
        received = True
        return {'body': serialized_body, 'type': 'http.request', 'more_body': False}
    return mock_receive