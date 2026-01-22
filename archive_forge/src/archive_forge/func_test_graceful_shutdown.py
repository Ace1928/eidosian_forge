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
def test_graceful_shutdown(self):
    svc = _Service()
    launcher = service.launch(self.conf, svc)
    svc.init.wait()
    launcher.stop()
    self.assertTrue(svc.cleaned_up)
    launcher.stop()