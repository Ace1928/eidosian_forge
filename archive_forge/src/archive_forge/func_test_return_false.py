import time
from unittest import mock
import eventlet
from eventlet.green import threading as greenthreading
from oslotest import base as test_base
from oslo_service import fixture
from oslo_service import loopingcall
def test_return_false(self):

    def _raise_it():
        raise loopingcall.LoopingCallDone(False)
    timer = loopingcall.DynamicLoopingCall(_raise_it)
    self.assertFalse(timer.start().wait())