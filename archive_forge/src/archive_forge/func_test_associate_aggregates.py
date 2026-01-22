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
def test_associate_aggregates(self):
    self.placement_api_client.associate_aggregates(RESOURCE_PROVIDER_UUID, mock.ANY)
    self.placement_fixture.mock_put.assert_called_once_with('/resource_providers/%s/aggregates' % RESOURCE_PROVIDER_UUID, mock.ANY)