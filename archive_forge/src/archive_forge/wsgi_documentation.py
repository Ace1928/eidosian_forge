import io
import math
import sys
import typing
import warnings
import anyio
from anyio.abc import ObjectReceiveStream, ObjectSendStream
from starlette.types import Receive, Scope, Send

    Builds a scope and request body into a WSGI environ object.
    