import uuid
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import mapping_fixtures
from keystone.tests.unit import test_v3
def test_federation_protocol_duplicate_conflict_gives_name(self):
    self.idp = {'id': uuid.uuid4().hex, 'enabled': True, 'description': uuid.uuid4().hex}
    PROVIDERS.federation_api.create_idp(self.idp['id'], self.idp)
    self.mapping = mapping_fixtures.MAPPING_EPHEMERAL_USER
    self.mapping['id'] = uuid.uuid4().hex
    PROVIDERS.federation_api.create_mapping(self.mapping['id'], self.mapping)
    protocol = {'id': uuid.uuid4().hex, 'mapping_id': self.mapping['id']}
    protocol_ret = PROVIDERS.federation_api.create_protocol(self.idp['id'], protocol['id'], protocol)
    try:
        PROVIDERS.federation_api.create_protocol(self.idp['id'], protocol['id'], protocol)
    except exception.Conflict as e:
        self.assertIn('Duplicate entry found with ID %s' % protocol_ret['id'], repr(e))
    else:
        self.fail('Create duplicate federation_protocol did not raise a conflict')