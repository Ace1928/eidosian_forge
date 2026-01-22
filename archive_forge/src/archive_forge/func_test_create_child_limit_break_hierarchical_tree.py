import http.client
import uuid
from keystone.common import provider_api
from keystone.common.validation import validators
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_create_child_limit_break_hierarchical_tree(self):
    ref = unit.new_limit_ref(domain_id=self.domain_A['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=20)
    self.post('/limits', body={'limits': [ref]}, token=self.system_admin_token, expected_status=http.client.CREATED)
    ref = unit.new_limit_ref(project_id=self.project_B['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=15)
    self.post('/limits', body={'limits': [ref]}, token=self.system_admin_token, expected_status=http.client.CREATED)
    ref = unit.new_limit_ref(project_id=self.project_C['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=21)
    self.post('/limits', body={'limits': [ref]}, token=self.system_admin_token, expected_status=http.client.FORBIDDEN)