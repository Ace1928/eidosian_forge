import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_region_service_association_cleanup_when_region_deleted(self):
    url = '/policies/%(policy_id)s/OS-ENDPOINT-POLICY/services/%(service_id)s/regions/%(region_id)s' % {'policy_id': self.policy['id'], 'service_id': self.service['id'], 'region_id': self.region['id']}
    self.put(url)
    self.head(url)
    self.delete('/regions/%(region_id)s' % {'region_id': self.region['id']})
    self.head(url, expected_status=http.client.NOT_FOUND)