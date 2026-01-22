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
def test_mutate_hook_process_launcher(self):
    """Test mutate_config_files is called by ProcessLauncher on SIGHUP.

        Forks happen in _spawn_service and ProcessLauncher. So we get three
        tiers of processes, the top tier being the test process. self.pid
        refers to the middle tier, which represents our application. Both
        service_maker and launcher_maker execute in the middle tier. The bottom
        tier is the workers.

        The behavior we want is that when the application (middle tier)
        receives a SIGHUP, it catches that, calls mutate_config_files and
        relaunches all the workers. This causes them to inherit the mutated
        config.
        """
    mutate = multiprocessing.Event()
    ready = multiprocessing.Event()

    def service_maker():
        self.conf.register_mutate_hook(lambda c, f: mutate.set())
        return ServiceWithTimer(ready)

    def launcher_maker():
        return service.ProcessLauncher(self.conf, restart_method='mutate')
    self.pid = self._spawn_service(1, service_maker, launcher_maker)
    timeout = 5
    ready.wait(timeout)
    self.assertTrue(ready.is_set(), 'Service never became ready')
    ready.clear()
    self.assertFalse(mutate.is_set(), 'Hook was called too early')
    os.kill(self.pid, signal.SIGHUP)
    ready.wait(timeout)
    self.assertTrue(ready.is_set(), 'Service never back after SIGHUP')
    self.assertTrue(mutate.is_set(), "Hook wasn't called")