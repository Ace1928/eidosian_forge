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
def dealer_router(create_bound_pair):
    return create_bound_pair(zmq.DEALER, zmq.ROUTER)