import uuid
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
def test_delete_limit_project(self):
    limit_1 = unit.new_limit_ref(project_id=self.project_bar['id'], service_id=self.service_one['id'], region_id=self.region_one['id'], resource_name='volume', resource_limit=10, id=uuid.uuid4().hex)
    limit_2 = unit.new_limit_ref(project_id=self.project_bar['id'], service_id=self.service_one['id'], region_id=self.region_two['id'], resource_name='snapshot', resource_limit=5, id=uuid.uuid4().hex)
    PROVIDERS.unified_limit_api.create_limits([limit_1, limit_2])
    PROVIDERS.resource_api.delete_project(self.project_baz['id'])
    ref = PROVIDERS.unified_limit_api.list_limits()
    self.assertEqual(2, len(ref))
    PROVIDERS.resource_api.delete_project(self.project_bar['id'])
    ref = PROVIDERS.unified_limit_api.list_limits()
    self.assertEqual([], ref)