import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit import test_v3
def test_create_endpoint_with_region(self):
    """EndpointV3 creates the region before creating the endpoint.

        This occurs when endpoint is provided with 'region' and no 'region_id'.
        """
    ref = unit.new_endpoint_ref_with_region(service_id=self.service_id, region=uuid.uuid4().hex)
    self.post('/endpoints', body={'endpoint': ref})
    self.get('/regions/%(region_id)s' % {'region_id': ref['region']})