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
def test_connectbind_context_managers(self):
    url = 'inproc://a'
    msg = b'hi'
    with self.Context() as ctx:
        with ctx.socket(zmq.PUSH) as a, ctx.socket(zmq.PULL) as b:
            a.bind(url)
            connect_context = b.connect(url)
            assert f'connect={url!r}' in repr(connect_context)
            with connect_context:
                a.send(msg)
                rcvd = self.recv(b)
                assert rcvd == msg
            with pytest.raises(zmq.Again):
                a.send(msg, flags=zmq.DONTWAIT)
            with pytest.raises(zmq.Again):
                b.recv(flags=zmq.DONTWAIT)
            a.unbind(url)
        with ctx.socket(zmq.PUSH) as a, ctx.socket(zmq.PULL) as b:
            bind_context = a.bind(url)
            assert f'bind={url!r}' in repr(bind_context)
            with bind_context:
                b.connect(url)
                a.send(msg)
                rcvd = self.recv(b)
                assert rcvd == msg
                b.disconnect(url)
            b.connect(url)
            with pytest.raises(zmq.Again):
                a.send(msg, flags=zmq.DONTWAIT)
            with pytest.raises(zmq.Again):
                b.recv(flags=zmq.DONTWAIT)