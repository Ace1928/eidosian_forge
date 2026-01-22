import datetime
import ssl
import sys
import threading
import time
import uuid
import fixtures
import kombu
import kombu.connection
import kombu.transport.memory
from oslo_serialization import jsonutils
from oslo_utils import eventletutils
import testscenarios
import oslo_messaging
from oslo_messaging._drivers import amqpdriver
from oslo_messaging._drivers import common as driver_common
from oslo_messaging._drivers import impl_rabbit as rabbit_driver
from oslo_messaging.exceptions import ConfigurationError
from oslo_messaging.exceptions import MessageDeliveryFailure
from oslo_messaging.tests import utils as test_utils
from oslo_messaging.transport import DriverLoadFailure
from unittest import mock
def test_consume_timeout(self):
    transport = oslo_messaging.get_transport(self.conf, 'kombu+memory:////')
    self.addCleanup(transport.cleanup)
    deadline = time.time() + 6
    with transport._driver._get_connection(driver_common.PURPOSE_LISTEN) as conn:
        self.assertRaises(driver_common.Timeout, conn.consume, timeout=3)
        conn.connection.connection.recoverable_channel_errors = (IOError,)
        conn.declare_fanout_consumer('notif.info', lambda msg: True)
        with mock.patch('kombu.connection.Connection.drain_events', side_effect=IOError):
            self.assertRaises(driver_common.Timeout, conn.consume, timeout=3)
    self.assertEqual(0, int(deadline - time.time()))