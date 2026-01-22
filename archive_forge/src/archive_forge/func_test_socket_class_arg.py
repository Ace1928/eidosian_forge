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
def test_socket_class_arg(self):

    class CustomSocket(zmq.Socket):
        pass
    with self.Context() as ctx:
        with ctx.socket(zmq.PUSH, socket_class=CustomSocket) as s:
            assert isinstance(s, CustomSocket)