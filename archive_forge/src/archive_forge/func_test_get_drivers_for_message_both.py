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
def test_get_drivers_for_message_both(self):
    config = '\ngroup_1:\n   rpc:\n       accepted_priorities:\n          - info\n       accepted_events:\n          - foo.*\n   driver_1:\n       accepted_priorities:\n          - info\n   driver_2:\n      accepted_events:\n          - foo.*\n        '
    groups = yaml.safe_load(config)
    group = groups['group_1']
    self.assertEqual(['driver_2'], self.router._get_drivers_for_message(group, 'foo.blah', 'unknown'))
    self.assertEqual(['driver_1'], self.router._get_drivers_for_message(group, 'unknown', 'info'))
    x = self.router._get_drivers_for_message(group, 'foo.blah', 'info')
    x.sort()
    self.assertEqual(['driver_1', 'driver_2', 'rpc'], x)