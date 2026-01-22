import uuid
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
def test_delete_limit(self):
    limit_1 = unit.new_limit_ref(project_id=self.project_bar['id'], service_id=self.service_one['id'], region_id=self.region_one['id'], resource_name='volume', resource_limit=10, id=uuid.uuid4().hex)
    limit_2 = unit.new_limit_ref(project_id=self.project_bar['id'], service_id=self.service_one['id'], region_id=self.region_two['id'], resource_name='snapshot', resource_limit=5, id=uuid.uuid4().hex)
    PROVIDERS.unified_limit_api.create_limits([limit_1, limit_2])
    PROVIDERS.unified_limit_api.delete_limit(limit_1['id'])
    self.assertRaises(exception.LimitNotFound, PROVIDERS.unified_limit_api.get_limit, limit_1['id'])