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
def test_load_notifiers_config(self):
    self.config(routing_config='routing_notifier.yaml', group='oslo_messaging_notifications')
    routing_config = '\ngroup_1:\n   rpc : foo\ngroup_2:\n   rpc : blah\n        '
    config_file = mock.MagicMock()
    config_file.return_value = routing_config
    with mock.patch.object(self.router, '_get_notifier_config_file', config_file):
        with mock.patch('stevedore.dispatch.DispatchExtensionManager', return_value=self._fake_extension_manager(mock.MagicMock())):
            with mock.patch('oslo_messaging.notify._impl_routing.LOG'):
                self.router._load_notifiers()
                groups = list(self.router.routing_groups.keys())
                groups.sort()
                self.assertEqual(['group_1', 'group_2'], groups)