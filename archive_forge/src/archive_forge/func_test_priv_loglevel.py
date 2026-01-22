import copy
import eventlet
import fixtures
import functools
import logging as pylogging
import platform
import sys
import time
from unittest import mock
from oslo_log import formatters
from oslo_log import log as logging
from oslotest import base
import testtools
from oslo_privsep import capabilities
from oslo_privsep import comm
from oslo_privsep import daemon
from oslo_privsep.tests import testctx
def test_priv_loglevel(self):
    logger = self.useFixture(fixtures.FakeLogger(level=logging.INFO))
    logme(logging.DEBUG, 'test@DEBUG')
    logme(logging.WARN, 'test@WARN')
    time.sleep(0.1)
    self.assertNotIn('test@DEBUG', logger.output)
    self.assertIn('test@WARN', logger.output)