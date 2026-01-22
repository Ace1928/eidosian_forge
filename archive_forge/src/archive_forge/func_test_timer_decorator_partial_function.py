import asyncio
import functools
import random
import re
import socket
from datetime import timedelta
from unittest import SkipTest, mock
from statsd import StatsClient
from statsd import TCPStatsClient
from statsd import UnixSocketStatsClient
def test_timer_decorator_partial_function():
    """TCPStatsClient.timer can be used as decorator on a partial function."""
    cl = _tcp_client()
    foo = functools.partial(lambda x: x * x, 2)
    func = cl.timer('foo')(foo)
    eq_(4, func())
    _timer_check(cl._sock, 1, 'tcp', 'foo', 'ms|@0.1')