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
def test_call_monitor_ok(self):
    driver = amqp_driver.ProtonDriver(self.conf, self._broker_url)
    target = oslo_messaging.Target(topic='test-topic')
    listener = _CallMonitor(driver.listen(target, None, None)._poll_style_listener, delay=11, hb_count=100)
    rc = driver.send(target, {'context': True}, {'method': 'echo', 'id': '1'}, wait_for_reply=True, timeout=60, call_monitor_timeout=5)
    self.assertIsNotNone(rc)
    self.assertEqual('1', rc.get('correlation-id'))
    listener.join(timeout=30)
    self.assertFalse(listener.is_alive())
    driver.cleanup()