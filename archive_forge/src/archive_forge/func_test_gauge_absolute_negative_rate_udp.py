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
@mock.patch.object(random, 'random')
def test_gauge_absolute_negative_rate_udp(mock_random):
    """StatsClient.gauge works with absolute negative value and rate."""
    cl = _udp_client()
    _test_gauge_absolute_negative_rate(cl, 'udp', mock_random)