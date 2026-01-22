from unittest import mock
from openstack import exceptions
from heat.engine.clients.os import senlin as senlin_plugin
from heat.tests import common
from heat.tests import utils
def test_get_profile_id(self):
    mock_profile = mock.Mock(id='fake_profile_id')
    mock_get = self.patchobject(self.client, 'get_profile', return_value=mock_profile)
    ret = self.plugin.get_profile_id('fake_profile')
    self.assertEqual('fake_profile_id', ret)
    mock_get.assert_called_once_with('fake_profile')