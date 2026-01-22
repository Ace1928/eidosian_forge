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
def test_pipeline_timer_decorator_udp():
    """UDP Pipeline manager can be used as decorator."""
    cl = _udp_client()
    _test_pipeline_timer_decorator(cl, 'udp')