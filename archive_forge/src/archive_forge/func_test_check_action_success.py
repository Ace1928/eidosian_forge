from unittest import mock
from openstack import exceptions
from heat.engine.clients.os import senlin as senlin_plugin
from heat.tests import common
from heat.tests import utils
def test_check_action_success(self):
    mock_action = mock.MagicMock()
    mock_action.status = 'SUCCEEDED'
    mock_get = self.patchobject(self.client, 'get_action')
    mock_get.return_value = mock_action
    self.assertTrue(self.plugin.check_action_status('fake_id'))
    mock_get.assert_called_once_with('fake_id')