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
def test_routable_address(self):
    self.config(addressing_mode='routable', group='oslo_messaging_amqp')
    _opts = self.conf.oslo_messaging_amqp
    notifications = [(oslo_messaging.Target(topic='test-topic'), 'info'), (oslo_messaging.Target(topic='test-topic'), 'error'), (oslo_messaging.Target(topic='test-topic'), 'debug')]
    msgs = self._address_test(oslo_messaging.Target(exchange='ex', topic='test-topic'), notifications)
    addrs = [m.address for m in msgs]
    notify_addrs = [a for a in addrs if a.startswith(_opts.notify_address_prefix)]
    self.assertEqual(len(notify_addrs), len(notifications))
    self.assertEqual(len(notifications), len([a for a in notify_addrs if _opts.anycast_address in a]))
    rpc_addrs = [a for a in addrs if a.startswith(_opts.rpc_address_prefix)]
    self.assertEqual(2, len([a for a in rpc_addrs if _opts.anycast_address in a]))
    self.assertEqual(1, len([a for a in rpc_addrs if _opts.multicast_address in a]))
    self.assertEqual(2, len([a for a in rpc_addrs if _opts.unicast_address in a]))