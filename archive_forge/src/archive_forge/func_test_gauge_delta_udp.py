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
def test_gauge_delta_udp():
    """StatsClient.gauge works with delta values."""
    cl = _udp_client()
    _test_gauge_delta(cl, 'udp')