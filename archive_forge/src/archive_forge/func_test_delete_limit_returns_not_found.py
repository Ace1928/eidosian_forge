import uuid
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
def test_delete_limit_returns_not_found(self):
    self.assertRaises(exception.LimitNotFound, PROVIDERS.unified_limit_api.delete_limit, uuid.uuid4().hex)