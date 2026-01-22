import logging
import multiprocessing
import os
import signal
import socket
import time
import traceback
from unittest import mock
import eventlet
from eventlet import event
from oslotest import base as test_base
from oslo_service import service
from oslo_service.tests import base
from oslo_service.tests import eventlet_service
def test__handle_signal(self):
    signal_handler = service.SignalHandler()
    signal_handler.clear()
    self.assertEqual(0, len(signal_handler._signal_handlers[signal.SIGTERM]))
    call_1, call_2 = (mock.Mock(), mock.Mock())
    signal_handler.add_handler('SIGTERM', call_1)
    signal_handler.add_handler('SIGTERM', call_2)
    self.assertEqual(2, len(signal_handler._signal_handlers[signal.SIGTERM]))
    signal_handler._handle_signal(signal.SIGTERM, 'test')
    time.sleep(0)
    for m in signal_handler._signal_handlers[signal.SIGTERM]:
        m.assert_called_once_with(signal.SIGTERM, 'test')
    signal_handler.clear()