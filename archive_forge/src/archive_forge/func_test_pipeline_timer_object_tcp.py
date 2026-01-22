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
def test_pipeline_timer_object_tcp():
    """Timer from TCP Pipeline manager works."""
    cl = _tcp_client()
    _test_pipeline_timer_object(cl, 'tcp')