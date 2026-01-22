import asyncio
import inspect
import os
import signal
import time
from functools import partial
from threading import Thread
import pytest
import zmq
import zmq.asyncio
def assert_raises_errno(errno):
    try:
        yield
    except zmq.ZMQError as e:
        assert e.errno == errno, f'wrong error raised, expected {zmq.ZMQError(errno)} got {zmq.ZMQError(e.errno)}'
    else:
        pytest.fail(f'Expected {zmq.ZMQError(errno)}, no error raised')