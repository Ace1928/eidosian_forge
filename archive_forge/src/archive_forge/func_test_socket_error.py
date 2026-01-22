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
def test_socket_error():
    """Socket error on StatsClient should be ignored."""
    cl = _udp_client()
    cl._sock.sendto.side_effect = socket.timeout()
    cl.incr('foo')
    _sock_check(cl._sock, 1, 'udp', 'foo:1|c')