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
def test_hwm(self):
    zmq3 = zmq.zmq_version_info()[0] >= 3
    for stype in (zmq.PUB, zmq.ROUTER, zmq.SUB, zmq.REQ, zmq.DEALER):
        s = self.context.socket(stype)
        s.hwm = 100
        assert s.hwm == 100
        if zmq3:
            try:
                assert s.sndhwm == 100
            except AttributeError:
                pass
            try:
                assert s.rcvhwm == 100
            except AttributeError:
                pass
        s.close()