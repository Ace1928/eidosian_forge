import copy
import errno
import json
import os
import platform
import socket
import sys
import time
import warnings
from unittest import mock
import pytest
from pytest import mark
import zmq
from zmq.tests import BaseZMQTestCase, GreenTest, SkipTest, have_gevent, skip_pypy
def test_poll(self):
    a, b = self.create_bound_pair()
    time.time()
    evt = a.poll(POLL_TIMEOUT)
    assert evt == 0
    evt = a.poll(POLL_TIMEOUT, zmq.POLLOUT)
    assert evt == zmq.POLLOUT
    msg = b'hi'
    a.send(msg)
    evt = b.poll(POLL_TIMEOUT)
    assert evt == zmq.POLLIN
    msg2 = self.recv(b)
    evt = b.poll(POLL_TIMEOUT)
    assert evt == 0
    assert msg2 == msg