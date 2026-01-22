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
def test_format_record(self):
    logs = []
    self.useFixture(fixtures.FakeLogger(level=logging.INFO, format='dummy', formatter=functools.partial(LogRecorder, logs)))
    logme(logging.WARN, 'test with exc', exc_info=True)
    time.sleep(0.1)
    self.assertEqual(1, len(logs))
    record = logs[0]
    fake_config = mock.Mock(logging_default_format_string='NOCTXT: %(message)s')
    formatter = formatters.ContextFormatter(config=fake_config)
    formatter.format(record)