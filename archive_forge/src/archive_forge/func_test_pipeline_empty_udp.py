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
def test_pipeline_empty_udp():
    """Pipelines should be empty after a send() call (UDP)."""
    cl = _udp_client()
    _test_pipeline_empty(cl)