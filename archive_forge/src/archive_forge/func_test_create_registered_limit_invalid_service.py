import uuid
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
def test_create_registered_limit_invalid_service(self):
    registered_limit_1 = unit.new_registered_limit_ref(service_id=uuid.uuid4().hex, region_id=self.region_one['id'], resource_name='volume', default_limit=10, id=uuid.uuid4().hex)
    self.assertRaises(exception.ValidationError, PROVIDERS.unified_limit_api.create_registered_limits, [registered_limit_1])