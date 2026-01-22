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
@mock.patch('oslo_service.service.ProcessLauncher._start_child')
@mock.patch('oslo_service.service.ProcessLauncher.handle_signal')
@mock.patch('eventlet.greenio.GreenPipe')
@mock.patch('os.pipe')
def test_double_sighup(self, pipe_mock, green_pipe_mock, handle_signal_mock, start_child_mock):
    pipe_mock.return_value = [None, None]
    launcher = service.ProcessLauncher(self.conf)
    serv = _Service()
    launcher.launch_service(serv, workers=0)

    def stager():
        stager.stage += 1
        if stager.stage < 3:
            launcher._handle_hup(1, mock.sentinel.frame)
        elif stager.stage == 3:
            launcher._handle_term(15, mock.sentinel.frame)
        else:
            self.fail('TERM did not kill launcher')
    stager.stage = -1
    handle_signal_mock.side_effect = stager
    launcher.wait()
    self.assertEqual(3, stager.stage)