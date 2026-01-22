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
@mock.patch('signal.alarm')
@mock.patch('oslo_service.service.ProcessLauncher.launch_service')
def test_multiple_worker(self, mock_launch, alarm_mock):
    svc = service.Service()
    service.launch(self.conf, svc, workers=3)
    mock_launch.assert_called_with(svc, workers=3)