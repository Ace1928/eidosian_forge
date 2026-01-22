import uuid
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
def test_registering_unsupported_enforcement_model_fails(self):
    self.assertRaises(ValueError, self.config_fixture.config, group='unified_limit', enforcement_model=uuid.uuid4().hex)