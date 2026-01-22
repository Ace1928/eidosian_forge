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
def test_child_signal_sighup(self):
    start_workers = self._spawn()
    os.kill(start_workers[0], signal.SIGHUP)
    cond = lambda: start_workers != self._get_workers()
    timeout = 5
    self._wait(cond, timeout)
    end_workers = self._get_workers()
    LOG.info('workers: %r' % end_workers)
    self.assertEqual(start_workers, end_workers)