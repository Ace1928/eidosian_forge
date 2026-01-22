import asyncio
import logging
import os
import shutil
import sys
import warnings
from contextlib import contextmanager
import pytest
import zmq
import zmq.asyncio
import zmq.auth
from zmq.tests import SkipTest, skip_pypy
@contextmanager
def push_pull(self):
    with self.context.socket(zmq.PUSH) as server, self.context.socket(zmq.PULL) as client:
        server.linger = 0
        server.sndtimeo = 2000
        client.linger = 0
        client.rcvtimeo = 2000
        yield (server, client)