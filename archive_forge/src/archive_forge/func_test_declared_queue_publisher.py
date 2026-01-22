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
def test_declared_queue_publisher(self):
    transport = oslo_messaging.get_transport(self.conf, 'kombu+memory:////')
    self.addCleanup(transport.cleanup)
    e_passive = kombu.entity.Exchange(name='foobar', type='topic', passive=True)
    e_active = kombu.entity.Exchange(name='foobar', type='topic', passive=False)
    with transport._driver._get_connection(driver_common.PURPOSE_SEND) as pool_conn:
        conn = pool_conn.connection
        exc = conn.connection.channel_errors[0]

        def try_send(exchange):
            conn._ensure_publishing(conn._publish_and_creates_default_queue, exchange, {}, routing_key='foobar')
        with mock.patch('kombu.transport.virtual.Channel.close'):
            self.assertRaises(oslo_messaging.MessageDeliveryFailure, try_send, e_passive)
            try_send(e_active)
            try_send(e_passive)
        with mock.patch('kombu.messaging.Producer.publish', side_effect=exc):
            with mock.patch('kombu.transport.virtual.Channel.close'):
                self.assertIn('foobar', conn._declared_exchanges)
                self.assertRaises(oslo_messaging.MessageDeliveryFailure, try_send, e_passive)
                self.assertEqual(0, len(conn._declared_exchanges))
        try_send(e_active)
        self.assertIn('foobar', conn._declared_exchanges)