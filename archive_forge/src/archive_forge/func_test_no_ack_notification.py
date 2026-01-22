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
def test_no_ack_notification(self):
    """Verify no exception is thrown if acks are turned off"""
    self.config(pre_settled=['notify', 'fleabag', 'poochie'], group='oslo_messaging_amqp')
    driver = amqp_driver.ProtonDriver(self.conf, self._broker_url)
    driver._default_notify_timeout = 2
    driver.send_notification(oslo_messaging.Target(topic='!no-ack!'), 'context', {'target': '!no-ack!'}, 2.0)
    driver.cleanup()