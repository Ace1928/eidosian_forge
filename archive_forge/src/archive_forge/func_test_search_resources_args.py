from unittest import mock
from openstack import exceptions
from openstack import proxy
from openstack import resource
from openstack.tests.unit import base
def test_search_resources_args(self):
    self.session._get.side_effect = exceptions.ResourceNotFound
    self.session._list.return_value = []
    self.cloud.search_resources('mock_session.fake', 'fake_name', get_args=['getarg1'], get_kwargs={'getkwarg1': '1'}, list_args=['listarg1'], list_kwargs={'listkwarg1': '1'}, filter1='foo')
    self.session._get.assert_called_with(self.FakeResource, 'fake_name', 'getarg1', getkwarg1='1')
    self.session._list.assert_called_with(self.FakeResource, 'listarg1', listkwarg1='1', name='fake_name', filter1='foo')