import uuid
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import mapping_fixtures
from keystone.tests.unit import test_v3
def test_mapping_duplicate_conflict_with_id_in_id(self):
    self.mapping = mapping_fixtures.MAPPING_EPHEMERAL_USER
    self.mapping['id'] = 'mapping_with_id_in_the_id'
    PROVIDERS.federation_api.create_mapping(self.mapping['id'], self.mapping)
    try:
        PROVIDERS.federation_api.create_mapping(self.mapping['id'], self.mapping)
    except exception.Conflict as e:
        self.assertIn('Duplicate entry found with ID %s' % self.mapping['id'], repr(e))