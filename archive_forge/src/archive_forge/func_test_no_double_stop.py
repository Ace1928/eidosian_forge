import time
from unittest import mock
import eventlet
from eventlet.green import threading as greenthreading
from oslotest import base as test_base
from oslo_service import fixture
from oslo_service import loopingcall
def test_no_double_stop(self):

    def _raise_it():
        raise loopingcall.LoopingCallDone(False)
    timer = loopingcall.FixedIntervalLoopingCall(_raise_it)
    timer.start(interval=0.5)
    timer.stop()
    timer.stop()