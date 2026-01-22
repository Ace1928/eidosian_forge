import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit import test_v3
def test_deleting_endpoint_with_space_in_url(self):
    url_with_space = 'http://127.0.0.1:8774 /v1.1/\\$(tenant_i d)s'
    ref = unit.new_endpoint_ref(service_id=self.service['id'], region_id=None, publicurl=url_with_space, internalurl=url_with_space, adminurl=url_with_space, url=url_with_space)
    PROVIDERS.catalog_api.create_endpoint(ref['id'], ref)
    self.delete('/endpoints/%s' % ref['id'])
    self.get('/endpoints/%s' % ref['id'], expected_status=http.client.NOT_FOUND)