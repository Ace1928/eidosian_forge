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
@mock.patch.object(socket, 'socket')
def test_tcp_raises_exception_to_user(mock_socket):
    """Socket errors in TCPStatsClient should be raised to user."""
    addr = ('127.0.0.1', 1234)
    cl = _tcp_client(addr=addr[0], port=addr[1])
    cl.incr('foo')
    eq_(1, cl._sock.sendall.call_count)
    cl._sock.sendall.side_effect = socket.error
    with assert_raises(socket.error):
        cl.incr('foo')