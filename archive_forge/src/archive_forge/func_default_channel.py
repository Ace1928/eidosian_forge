from __future__ import annotations
import os
import socket
import sys
from contextlib import contextmanager
from itertools import count, cycle
from operator import itemgetter
from typing import TYPE_CHECKING, Any
from kombu import exceptions
from .log import get_logger
from .resource import Resource
from .transport import get_transport_cls, supports_librabbitmq
from .utils.collections import HashedSeq
from .utils.functional import dictfilter, lazy, retry_over_time, shufflecycle
from .utils.objects import cached_property
from .utils.url import as_url, maybe_sanitize_url, parse_url, quote, urlparse
@property
def default_channel(self) -> Channel:
    """Default channel.

        Created upon access and closed when the connection is closed.

        Note:
        ----
            Can be used for automatic channel handling when you only need one
            channel, and also it is the channel implicitly used if
            a connection is passed instead of a channel, to functions that
            require a channel.
        """
    conn_opts = self._extract_failover_opts()
    self._ensure_connection(**conn_opts)
    if self._default_channel is None:
        self._default_channel = self.channel()
    return self._default_channel