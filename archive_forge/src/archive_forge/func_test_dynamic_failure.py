import contextlib
import logging
import threading
import time
from oslo_serialization import jsonutils
from oslo_utils import reflection
from zake import fake_client
import taskflow.engines
from taskflow import exceptions as exc
from taskflow.jobs import backends as jobs
from taskflow.listeners import claims
from taskflow.listeners import logging as logging_listeners
from taskflow.listeners import timing
from taskflow.patterns import linear_flow as lf
from taskflow.persistence.backends import impl_memory
from taskflow import states
from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils as test_utils
from taskflow.utils import misc
from taskflow.utils import persistence_utils
def test_dynamic_failure(self):
    flow = lf.Flow('test')
    flow.add(test_utils.TaskWithFailure('test-1'))
    e = self._make_engine(flow)
    log, handler = self._make_logger()
    with logging_listeners.DynamicLoggingListener(e, log=log):
        self.assertRaises(RuntimeError, e.run)
    self.assertGreater(0, handler.counts[logging.WARNING])
    self.assertGreater(0, handler.counts[logging.DEBUG])
    self.assertEqual(1, len(handler.exc_infos))
    for levelno in _LOG_LEVELS - set([logging.DEBUG, logging.WARNING]):
        self.assertEqual(0, handler.counts[levelno])