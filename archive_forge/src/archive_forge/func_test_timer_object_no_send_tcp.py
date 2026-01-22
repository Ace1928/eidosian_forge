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
def test_timer_object_no_send_tcp():
    """Stop TCPStatsClient.timer without sending."""
    cl = _tcp_client()
    _test_timer_object_no_send(cl, 'tcp')