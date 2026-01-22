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
def test_pipeline_packet_size():
    """Pipelines shouldn't send packets larger than 512 bytes (UDP only)."""
    sc = _udp_client()
    pipe = sc.pipeline()
    for x in range(32):
        pipe.incr('sixteen_char_str')
    pipe.send()
    eq_(2, sc._sock.sendto.call_count)
    assert len(sc._sock.sendto.call_args_list[0][0][0]) <= 512
    assert len(sc._sock.sendto.call_args_list[1][0][0]) <= 512