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
def test_crashed_service(self):
    service_maker = lambda: ServiceCrashOnStart()
    self.pid = self._spawn_service(service_maker=service_maker)
    status = self._reap_test()
    self.assertTrue(os.WIFEXITED(status))
    self.assertEqual(1, os.WEXITSTATUS(status))