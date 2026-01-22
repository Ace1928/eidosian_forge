import copy
import logging
import os
import queue
import select
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from unittest import mock
import uuid
from oslo_utils import eventletutils
from oslo_utils import importutils
from string import Template
import testtools
import oslo_messaging
from oslo_messaging.tests import utils as test_utils
def test_call_failed_reply(self):
    """Send back an exception generated at the listener"""

    class _FailedResponder(_ListenerThread):

        def __init__(self, listener):
            super(_FailedResponder, self).__init__(listener, 1)

        def run(self):
            self.started.set()
            while not self._done.is_set():
                for in_msg in self.listener.poll(timeout=0.5):
                    try:
                        raise RuntimeError('Oopsie!')
                    except RuntimeError:
                        in_msg.reply(reply=None, failure=sys.exc_info())
                    self._done.set()
    driver = amqp_driver.ProtonDriver(self.conf, self._broker_url)
    target = oslo_messaging.Target(topic='test-topic')
    listener = _FailedResponder(driver.listen(target, None, None)._poll_style_listener)
    self.assertRaises(RuntimeError, driver.send, target, {'context': 'whatever'}, {'method': 'echo'}, wait_for_reply=True, timeout=5.0)
    listener.join(timeout=30)
    self.assertFalse(listener.is_alive())
    driver.cleanup()