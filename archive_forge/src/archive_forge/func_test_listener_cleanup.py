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
def test_listener_cleanup(self):
    """Verify unused listener can cleanly shutdown."""
    driver = amqp_driver.ProtonDriver(self.conf, self._broker_url)
    target = oslo_messaging.Target(topic='test-topic')
    listener = driver.listen(target, None, None)._poll_style_listener
    self.assertIsInstance(listener, amqp_driver.ProtonListener)
    driver.cleanup()