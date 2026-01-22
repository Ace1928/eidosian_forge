import http.client
import uuid
from keystone.common import provider_api
from keystone.common.validation import validators
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_create_multi_limits_invalid_input(self):
    ref_A = unit.new_limit_ref(domain_id=self.domain_A['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=12)
    ref_B = unit.new_limit_ref(project_id=self.project_B['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=9)
    ref_C = unit.new_limit_ref(project_id=self.project_C['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=5)
    ref_D = unit.new_limit_ref(domain_id=self.domain_D['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=9)
    ref_E = unit.new_limit_ref(project_id=self.project_E['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=5)
    ref_F = unit.new_limit_ref(project_id=self.project_F['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=10)
    self.post('/limits', body={'limits': [ref_A, ref_B, ref_C, ref_D, ref_E, ref_F]}, token=self.system_admin_token, expected_status=http.client.FORBIDDEN)