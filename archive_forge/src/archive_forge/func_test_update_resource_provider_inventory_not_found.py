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
def test_update_resource_provider_inventory_not_found(self):
    self.placement_fixture.mock_put.side_effect = ks_exc.NotFound(details='No resource provider with uuid')
    self.assertRaises(n_exc.PlacementResourceNotFound, self.placement_api_client.update_resource_provider_inventory, RESOURCE_PROVIDER_UUID, INVENTORY, RESOURCE_CLASS_NAME, RESOURCE_PROVIDER_GENERATION)