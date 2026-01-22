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
def crash_zmq():
    ctx = self.Context()
    sock = ctx.socket(zmq.PULL)
    c = CyclicReference()
    c.crash(sock)
    ctx.destroy()