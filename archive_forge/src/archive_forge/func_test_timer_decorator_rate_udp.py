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
@mock.patch.object(random, 'random', lambda: -1)
def test_timer_decorator_rate_udp():
    """StatsClient.timer can be used as decorator with rate."""
    cl = _udp_client()
    _test_timer_decorator_rate(cl, 'udp')