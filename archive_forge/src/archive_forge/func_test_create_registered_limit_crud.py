import uuid
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
def test_create_registered_limit_crud(self):
    registered_limit_1 = unit.new_registered_limit_ref(service_id=self.service_one['id'], region_id=self.region_one['id'], resource_name='volume', default_limit=10, id=uuid.uuid4().hex, description='test description')
    reg_limits = PROVIDERS.unified_limit_api.create_registered_limits([registered_limit_1])
    self.assertDictEqual(registered_limit_1, reg_limits[0])
    registered_limit_2 = unit.new_registered_limit_ref(service_id=self.service_one['id'], region_id=self.region_one['id'], resource_name='snapshot', default_limit=5, id=uuid.uuid4().hex)
    registered_limit_3 = unit.new_registered_limit_ref(service_id=self.service_one['id'], region_id=self.region_one['id'], resource_name='backup', default_limit=5, id=uuid.uuid4().hex)
    reg_limits = PROVIDERS.unified_limit_api.create_registered_limits([registered_limit_2, registered_limit_3])
    self.assertEqual(2, len(reg_limits))
    for reg_limit in reg_limits:
        if reg_limit['id'] == registered_limit_2['id']:
            self.assertDictEqual(registered_limit_2, reg_limit)
        if reg_limit['id'] == registered_limit_3['id']:
            self.assertDictEqual(registered_limit_3, reg_limit)