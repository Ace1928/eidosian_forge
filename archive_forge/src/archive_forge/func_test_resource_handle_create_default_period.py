from unittest import mock
from heat.common import exception
from heat.engine.cfn import functions as cfn_funcs
from heat.engine.clients.os import monasca as client_plugin
from heat.engine.resources.openstack.monasca import notification
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_resource_handle_create_default_period(self):
    self.test_resource.properties.data.pop('period')
    mock_notification_create = self.test_client.notifications.create
    self.test_resource.handle_create()
    args = dict(name='test-notification', type='webhook', address='http://localhost:80/', period=60)
    mock_notification_create.assert_called_once_with(**args)