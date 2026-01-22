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
def maybe_channel(channel: Channel | Connection) -> Channel:
    """Get channel from object.

    Return the default channel if argument is a connection instance,
    otherwise just return the channel given.
    """
    if is_connection(channel):
        return channel.default_channel
    return channel