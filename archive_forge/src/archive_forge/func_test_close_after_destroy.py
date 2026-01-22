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
def test_close_after_destroy(self):
    """s.close() after ctx.destroy() should be fine"""
    ctx = self.Context()
    s = ctx.socket(zmq.REP)
    ctx.destroy()
    time.sleep(0.01)
    s.close()
    assert s.closed