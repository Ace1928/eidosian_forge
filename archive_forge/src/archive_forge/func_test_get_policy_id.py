from unittest import mock
from openstack import exceptions
from heat.engine.clients.os import senlin as senlin_plugin
from heat.tests import common
from heat.tests import utils
def test_get_policy_id(self):
    mock_policy = mock.Mock(id='fake_policy_id')
    mock_get = self.patchobject(self.client, 'get_policy', return_value=mock_policy)
    ret = self.plugin.get_policy_id('fake_policy')
    self.assertEqual('fake_policy_id', ret)
    mock_get.assert_called_once_with('fake_policy')