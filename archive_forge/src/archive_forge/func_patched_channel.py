from functools import wraps
import grpc
from grpc import Channel, Server, intercept_channel
from grpc.aio import Channel as AsyncChannel
from grpc.aio import Server as AsyncServer
from sentry_sdk.integrations import Integration
from sentry_sdk._types import TYPE_CHECKING
from .client import ClientInterceptor
from .server import ServerInterceptor
from .aio.server import ServerInterceptor as AsyncServerInterceptor
from .aio.client import (
from .aio.client import (
from typing import Any, Optional, Sequence
@wraps(func)
def patched_channel(*args: P.args, interceptors: Optional[Sequence[grpc.aio.ClientInterceptor]]=None, **kwargs: P.kwargs) -> Channel:
    sentry_interceptors = [AsyncUnaryUnaryClientInterceptor(), AsyncUnaryStreamClientIntercetor()]
    interceptors = [*sentry_interceptors, *(interceptors or [])]
    return func(*args, interceptors=interceptors, **kwargs)