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
def test_update_resource_provider_inventory_generation(self):
    expected_body = {'resource_provider_generation': RESOURCE_PROVIDER_GENERATION}
    expected_body.update(INVENTORY)
    self.placement_api_client.update_resource_provider_inventory(RESOURCE_PROVIDER_UUID, INVENTORY, RESOURCE_CLASS_NAME, resource_provider_generation=1)
    self.placement_fixture.mock_put.assert_called_once_with('/resource_providers/%(rp_uuid)s/inventories/%(rc_name)s' % {'rp_uuid': RESOURCE_PROVIDER_UUID, 'rc_name': RESOURCE_CLASS_NAME}, expected_body)