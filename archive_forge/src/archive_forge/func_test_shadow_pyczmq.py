import copy
import gc
import os
import sys
import time
from queue import Queue
from threading import Event, Thread
from unittest import mock
import pytest
from pytest import mark
import zmq
from zmq.tests import PYPY, BaseZMQTestCase, GreenTest, SkipTest
def test_shadow_pyczmq(self):
    try:
        from pyczmq import zctx, zsocket, zstr
    except Exception:
        raise SkipTest('Requires pyczmq')
    ctx = zctx.new()
    a = zsocket.new(ctx, zmq.PUSH)
    zsocket.bind(a, 'inproc://a')
    ctx2 = self.Context.shadow_pyczmq(ctx)
    b = ctx2.socket(zmq.PULL)
    b.connect('inproc://a')
    zstr.send(a, b'hi')
    rcvd = self.recv(b)
    assert rcvd == b'hi'
    b.close()