from types import SimpleNamespace
from typing import TYPE_CHECKING, Awaitable, Optional, Protocol, Type, TypeVar
import attr
from aiosignal import Signal
from multidict import CIMultiDict
from yarl import URL
from .client_reqrep import ClientResponse
@property
def on_request_start(self) -> 'Signal[_SignalCallback[TraceRequestStartParams]]':
    return self._on_request_start