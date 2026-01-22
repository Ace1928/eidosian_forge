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
def test_call_reply_timeout(self):
    """What happens if the replier times out?"""

    class _TimeoutListener(_ListenerThread):

        def __init__(self, listener):
            super(_TimeoutListener, self).__init__(listener, 1)

        def run(self):
            self.started.set()
            while not self._done.is_set():
                for in_msg in self.listener.poll(timeout=0.5):
                    in_msg._reply_to = '!no-ack!'
                    in_msg.reply(reply={'correlation-id': in_msg.message.get('id')})
                    self._done.set()
    driver = amqp_driver.ProtonDriver(self.conf, self._broker_url)
    driver._default_reply_timeout = 1
    target = oslo_messaging.Target(topic='test-topic')
    listener = _TimeoutListener(driver.listen(target, None, None)._poll_style_listener)
    self.assertRaises(oslo_messaging.MessagingTimeout, driver.send, target, {'context': 'whatever'}, {'method': 'echo'}, wait_for_reply=True, timeout=3)
    listener.join(timeout=30)
    self.assertFalse(listener.is_alive())
    driver.cleanup()