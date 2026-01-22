import uuid
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
def test_create_limit_description_none(self):
    limit = unit.new_limit_ref(project_id=self.project_bar['id'], service_id=self.service_one['id'], region_id=self.region_one['id'], resource_name='volume', resource_limit=10, id=uuid.uuid4().hex, description=None)
    res = PROVIDERS.unified_limit_api.create_limits([limit])
    self.assertIsNone(res[0]['description'])