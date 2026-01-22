import asyncio
import functools
import random
import sys
import traceback
import warnings
from collections import defaultdict, deque
from contextlib import suppress
from http import HTTPStatus
from http.cookies import SimpleCookie
from itertools import cycle, islice
from time import monotonic
from types import TracebackType
from typing import (
import attr
from . import hdrs, helpers
from .abc import AbstractResolver
from .client_exceptions import (
from .client_proto import ResponseHandler
from .client_reqrep import ClientRequest, Fingerprint, _merge_ssl_params
from .helpers import ceil_timeout, get_running_loop, is_ip_address, noop, sentinel
from .locks import EventResultOrError
from .resolver import DefaultResolver
def clear_dns_cache(self, host: Optional[str]=None, port: Optional[int]=None) -> None:
    """Remove specified host/port or clear all dns local cache."""
    if host is not None and port is not None:
        self._cached_hosts.remove((host, port))
    elif host is not None or port is not None:
        raise ValueError('either both host and port or none of them are allowed')
    else:
        self._cached_hosts.clear()