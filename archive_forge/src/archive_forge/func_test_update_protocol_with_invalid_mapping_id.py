import uuid
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit import mapping_fixtures
def test_update_protocol_with_invalid_mapping_id(self):
    protocol = {'id': uuid.uuid4().hex, 'mapping_id': self.mapping['id']}
    protocol_ret = PROVIDERS.federation_api.create_protocol(self.idp['id'], protocol['id'], protocol)
    self.assertEqual(protocol['id'], protocol_ret['id'])
    protocol['mapping_id'] = uuid.uuid4().hex
    self.assertRaises(exception.ValidationError, PROVIDERS.federation_api.update_protocol, self.idp['id'], protocol['id'], protocol)