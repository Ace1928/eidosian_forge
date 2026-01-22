import uuid
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
def test_update_registered_limit_when_reference_limit_exist(self):
    registered_limit_1 = unit.new_registered_limit_ref(service_id=self.service_one['id'], region_id=self.region_one['id'], resource_name='volume', default_limit=10, id=uuid.uuid4().hex)
    PROVIDERS.unified_limit_api.create_registered_limits([registered_limit_1])
    limit_1 = unit.new_limit_ref(project_id=self.project_bar['id'], service_id=self.service_one['id'], region_id=self.region_one['id'], resource_name='volume', resource_limit=10, id=uuid.uuid4().hex)
    PROVIDERS.unified_limit_api.create_limits([limit_1])
    registered_limit_update = {'id': registered_limit_1['id'], 'region_id': 'region_two'}
    self.assertRaises(exception.RegisteredLimitError, PROVIDERS.unified_limit_api.update_registered_limit, registered_limit_1['id'], registered_limit_update)
    registered_limit_2 = unit.new_registered_limit_ref(service_id=self.service_one['id'], resource_name='volume', default_limit=10, id=uuid.uuid4().hex)
    PROVIDERS.unified_limit_api.create_registered_limits([registered_limit_2])
    limit_2 = unit.new_limit_ref(project_id=self.project_bar['id'], service_id=self.service_one['id'], resource_name='volume', resource_limit=10, id=uuid.uuid4().hex)
    PROVIDERS.unified_limit_api.create_limits([limit_2])
    registered_limit_update = {'id': registered_limit_2['id'], 'region_id': 'region_two'}
    self.assertRaises(exception.RegisteredLimitError, PROVIDERS.unified_limit_api.update_registered_limit, registered_limit_2['id'], registered_limit_update)