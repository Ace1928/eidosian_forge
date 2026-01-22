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
def test_bad_attr(self):
    s = self.context.socket(zmq.DEALER)
    self.sockets.append(s)
    try:
        s.apple = 'foo'
    except AttributeError:
        pass
    else:
        self.fail('bad setattr should have raised AttributeError')
    try:
        s.apple
    except AttributeError:
        pass
    else:
        self.fail('bad getattr should have raised AttributeError')