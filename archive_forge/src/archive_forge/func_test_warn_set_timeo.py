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
@mark.skipif(not hasattr(zmq, 'RCVTIMEO'), reason='requires RCVTIMEO')
def test_warn_set_timeo(self):
    s = self.context.socket(zmq.REQ)
    with warnings.catch_warnings(record=True) as w:
        s.rcvtimeo = 5
    s.close()
    assert len(w) == 1
    assert w[0].category == UserWarning