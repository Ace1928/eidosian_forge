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
def test_notifier(self):
    self.config(driver=['noop'], group='oslo_messaging_notifications')
    transport = oslo_messaging.get_notification_transport(self.conf, url='fake:')
    notifier = oslo_messaging.Notifier(transport, 'test.localhost')
    self.assertFalse(notifier.is_enabled())