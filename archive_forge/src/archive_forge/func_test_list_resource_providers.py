from unittest import mock
from keystoneauth1 import exceptions as ks_exc
from keystoneauth1 import loading as keystone
from oslo_serialization import jsonutils
from oslo_utils import uuidutils
from neutron_lib._i18n import _
from neutron_lib.exceptions import placement as n_exc
from neutron_lib import fixture
from neutron_lib.placement import client as place_client
from neutron_lib.tests import _base as base
def test_list_resource_providers(self):
    filter_1 = {'name': 'name1', 'in_tree': 'tree1_uuid'}
    self.placement_api_client.list_resource_providers(**filter_1)
    args = str(self.placement_fixture.mock_get.call_args)
    self.placement_fixture.mock_get.assert_called_once()
    self.assertIn('name=name1', args)
    self.assertIn('in_tree=tree1_uuid', args)
    filter_2 = {'member_of': ['aggregate_uuid'], 'uuid': 'uuid_1', 'resources': {'r_class1': 'value1'}}
    self.placement_fixture.mock_get.reset_mock()
    self.placement_api_client.list_resource_providers(**filter_2)
    args = str(self.placement_fixture.mock_get.call_args)
    self.placement_fixture.mock_get.assert_called_once()
    self.assertIn('member_of', args)
    self.assertIn('uuid', args)
    self.assertIn('resources', args)
    filter_1.update(filter_2)
    self.placement_fixture.mock_get.reset_mock()
    self.placement_api_client.list_resource_providers(**filter_1)
    args = str(self.placement_fixture.mock_get.call_args)
    self.placement_fixture.mock_get.assert_called_once()
    for key in filter_1:
        self.assertIn(key, args)