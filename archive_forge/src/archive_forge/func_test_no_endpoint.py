import uuid
from keystoneauth1 import exceptions as ks_exc
import requests.exceptions
from openstack.config import cloud_region
from openstack import connection
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_no_endpoint(self):
    """Conf contains adapter opts, but service type not in catalog."""
    self.os_fixture.v3_token.remove_service('monitoring')
    conn = self._get_conn()
    self.assertRaises(ks_exc.catalog.EndpointNotFound, getattr, conn, 'monitoring')