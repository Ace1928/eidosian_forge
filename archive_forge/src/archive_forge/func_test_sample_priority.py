import datetime
import logging
import sys
import uuid
import fixtures
from kombu import connection
from oslo_serialization import jsonutils
from oslo_utils import strutils
from oslo_utils import timeutils
from stevedore import dispatch
from stevedore import extension
import testscenarios
import yaml
import oslo_messaging
from oslo_messaging.notify import _impl_log
from oslo_messaging.notify import _impl_test
from oslo_messaging.notify import messaging
from oslo_messaging.notify import notifier as msg_notifier
from oslo_messaging import serializer as msg_serializer
from oslo_messaging.tests import utils as test_utils
from unittest import mock
def test_sample_priority(self):
    driver = _impl_log.LogDriver(None, None, None)
    logger = mock.Mock(spec=logging.getLogger('oslo.messaging.notification.foo'))
    logger.sample = None
    msg = {'event_type': 'foo'}
    with mock.patch.object(logging, 'getLogger') as gl:
        gl.return_value = logger
        driver.notify(None, msg, 'sample', None)
        logging.getLogger.assert_called_once_with('oslo.messaging.notification.foo')