from unittest import mock
from heat.common import exception
from heat.engine.cfn import functions as cfn_funcs
from heat.engine.clients.os import monasca as client_plugin
from heat.engine.resources.openstack.monasca import notification
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_resource_handle_update_default_period(self):
    mock_notification_update = self.test_client.notifications.update
    self.test_resource.resource_id = '477e8273-60a7-4c41-b683-fdb0bc7cd151'
    self.test_resource.properties.data.pop('period')
    prop_diff = {notification.MonascaNotification.ADDRESS: 'http://localhost:1234/', notification.MonascaNotification.NAME: 'name-updated', notification.MonascaNotification.TYPE: 'webhook'}
    self.test_resource.handle_update(json_snippet=None, tmpl_diff=None, prop_diff=prop_diff)
    args = dict(notification_id=self.test_resource.resource_id, name='name-updated', type='webhook', address='http://localhost:1234/', period=60)
    mock_notification_update.assert_called_once_with(**args)