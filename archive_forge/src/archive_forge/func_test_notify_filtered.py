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
def test_notify_filtered(self):
    self.config(routing_config='routing_notifier.yaml', group='oslo_messaging_notifications')
    routing_config = '\ngroup_1:\n    rpc:\n        accepted_events:\n          - my_event\n    rpc2:\n        accepted_priorities:\n          - info\n    bar:\n        accepted_events:\n            - nothing\n        '
    config_file = mock.MagicMock()
    config_file.return_value = routing_config
    rpc_driver = mock.Mock()
    rpc2_driver = mock.Mock()
    bar_driver = mock.Mock()
    pm = dispatch.DispatchExtensionManager.make_test_instance([extension.Extension('rpc', None, None, rpc_driver), extension.Extension('rpc2', None, None, rpc2_driver), extension.Extension('bar', None, None, bar_driver)])
    with mock.patch.object(self.router, '_get_notifier_config_file', config_file):
        with mock.patch('stevedore.dispatch.DispatchExtensionManager', return_value=pm):
            with mock.patch('oslo_messaging.notify._impl_routing.LOG'):
                cxt = test_utils.TestContext()
                self.notifier.info(cxt, 'my_event', {})
                self.assertFalse(bar_driver.info.called)
                rpc_driver.notify.assert_called_once_with(cxt, mock.ANY, 'INFO', -1)
                rpc2_driver.notify.assert_called_once_with(cxt, mock.ANY, 'INFO', -1)