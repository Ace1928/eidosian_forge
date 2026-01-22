import asyncio
import logging
import warnings
from functools import partial, update_wrapper
from typing import (
from aiosignal import Signal
from frozenlist import FrozenList
from . import hdrs
from .abc import (
from .helpers import DEBUG, AppKey
from .http_parser import RawRequestMessage
from .log import web_logger
from .streams import StreamReader
from .typedefs import Middleware
from .web_exceptions import NotAppKeyWarning
from .web_log import AccessLogger
from .web_middlewares import _fix_request_current_app
from .web_protocol import RequestHandler
from .web_request import Request
from .web_response import StreamResponse
from .web_routedef import AbstractRouteDef
from .web_server import Server
from .web_urldispatcher import (
def pre_freeze(self) -> None:
    if self._pre_frozen:
        return
    self._pre_frozen = True
    self._middlewares.freeze()
    self._router.freeze()
    self._on_response_prepare.freeze()
    self._cleanup_ctx.freeze()
    self._on_startup.freeze()
    self._on_shutdown.freeze()
    self._on_cleanup.freeze()
    self._middlewares_handlers = tuple(self._prepare_middleware())
    self._run_middlewares = True if self.middlewares else False
    for subapp in self._subapps:
        subapp.pre_freeze()
        self._run_middlewares = self._run_middlewares or subapp._run_middlewares