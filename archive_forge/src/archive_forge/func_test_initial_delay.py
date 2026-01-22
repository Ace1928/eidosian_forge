import time
from unittest import mock
import eventlet
from eventlet.green import threading as greenthreading
from oslotest import base as test_base
from oslo_service import fixture
from oslo_service import loopingcall
@mock.patch('oslo_service.loopingcall.LoopingCallBase._sleep')
def test_initial_delay(self, sleep_mock):
    self.num_runs = 1
    timer = loopingcall.DynamicLoopingCall(self._wait_for_zero)
    timer.start(initial_delay=3).wait()
    sleep_mock.assert_has_calls([mock.call(3), mock.call(1)])