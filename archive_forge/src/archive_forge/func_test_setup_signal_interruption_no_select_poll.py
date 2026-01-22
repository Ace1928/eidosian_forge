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
def test_setup_signal_interruption_no_select_poll(self):
    service.SignalHandler.__class__._instances.clear()
    with mock.patch('eventlet.patcher.original', return_value=object()) as get_original:
        signal_handler = service.SignalHandler()
        get_original.assert_called_with('select')
    self.addCleanup(service.SignalHandler.__class__._instances.clear)
    self.assertFalse(signal_handler._SignalHandler__force_interrupt_on_signal)