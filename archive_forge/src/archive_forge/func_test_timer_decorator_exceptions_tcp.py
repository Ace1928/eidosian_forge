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
def test_timer_decorator_exceptions_tcp():
    cl = _tcp_client()
    _test_timer_decorator_exceptions(cl, 'tcp')