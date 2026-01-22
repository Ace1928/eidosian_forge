import os
import platform
import signal
import sys
import time
import warnings
from functools import partial
from threading import Thread
from typing import List
from unittest import SkipTest, TestCase
from pytest import mark
import zmq
from zmq.utils import jsonapi
def term_context(ctx, timeout):
    """Terminate a context with a timeout"""
    t = Thread(target=ctx.term)
    t.daemon = True
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        zmq.sugar.context.Context._instance = None
        raise RuntimeError('context could not terminate, open sockets likely remain in test')