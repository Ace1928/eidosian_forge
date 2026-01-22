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
@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
    assert dict(zmq.asyncio._selectors) == {}