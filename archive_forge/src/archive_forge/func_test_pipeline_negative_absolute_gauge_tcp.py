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
def test_pipeline_negative_absolute_gauge_tcp():
    """Negative absolute gauges use an internal pipeline (TCP)."""
    cl = _tcp_client()
    _test_pipeline_negative_absolute_gauge(cl, 'tcp')