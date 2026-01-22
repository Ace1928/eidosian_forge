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
def test_timer_context_exceptions_udp():
    cl = _udp_client()
    _test_timer_context_exceptions(cl, 'udp')