import uuid
from keystoneauth1 import exceptions as ks_exc
import requests.exceptions
from openstack.config import cloud_region
from openstack import connection
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_no_endpoint_ignore_service_type(self):
    """Bogus service type disabled if not in requested service_types."""
    self.assert_service_disabled('monitoring', 'Not in the list of requested service_types.', service_types={'compute', 'orchestration', 'bogus'})