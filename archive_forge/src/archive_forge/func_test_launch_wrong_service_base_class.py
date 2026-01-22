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
def test_launch_wrong_service_base_class(self):
    svc = mock.Mock()
    self.assertRaises(TypeError, service.launch, self.conf, svc)