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
def test_big_numbers_udp():
    """Test big numbers with UDP client."""
    cl = _udp_client()
    _test_big_numbers(cl, 'udp')