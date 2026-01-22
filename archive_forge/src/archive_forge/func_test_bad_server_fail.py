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
def test_bad_server_fail(self):
    self._ssl_config['s_cert'] = self._ssl_config['bad_cert']
    self._ssl_config['s_key'] = self._ssl_config['bad_key']
    self._broker = FakeBroker(self.conf.oslo_messaging_amqp, sock_addr=self._ssl_config['s_name'], ssl_config=self._ssl_config)
    url = oslo_messaging.TransportURL.parse(self.conf, 'amqp://%s:%d' % (self._broker.host, self._broker.port))
    self._broker.start()
    self.config(ssl_ca_file=self._ssl_config['ca_cert'], group='oslo_messaging_amqp')
    driver = amqp_driver.ProtonDriver(self.conf, url)
    target = oslo_messaging.Target(topic='test-topic')
    self.assertRaises(oslo_messaging.MessageDeliveryFailure, driver.send, target, {'context': 'whatever'}, {'method': 'echo', 'a': 'b'}, wait_for_reply=False, retry=1)
    driver.cleanup()