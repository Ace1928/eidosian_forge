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
def test_update_allocation(self):
    mock_rsp = mock.Mock()
    mock_rsp.json = lambda: {'allocations': {RESOURCE_PROVIDER_UUID: {'resources': {'a': 10}}}}
    self.placement_fixture.mock_get.side_effect = [mock_rsp]
    self.placement_api_client.update_allocation(CONSUMER_UUID, {'allocations': {RESOURCE_PROVIDER_UUID: {'resources': {'a': 20}}}})
    self.placement_fixture.mock_put.assert_called_once_with('/allocations/%s' % CONSUMER_UUID, {'allocations': {RESOURCE_PROVIDER_UUID: {'resources': {'a': 20}}}})