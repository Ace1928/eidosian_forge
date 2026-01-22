import uuid
from testtools import matchers
from keystone.common import provider_api
from keystone.common import sql
from keystone.identity.mapping_backends import mapping
from keystone.tests import unit
from keystone.tests.unit import identity_mapping as mapping_sql
from keystone.tests.unit import test_backend_sql
def test_id_mapping_handles_ids_greater_than_64_characters(self):
    initial_mappings = len(mapping_sql.list_id_mappings())
    local_id = 'Aa' * 100
    local_entity = {'domain_id': self.domainA['id'], 'local_id': local_id, 'entity_type': mapping.EntityType.GROUP}
    self.assertIsNone(PROVIDERS.id_mapping_api.get_public_id(local_entity))
    public_id = PROVIDERS.id_mapping_api.create_id_mapping(local_entity)
    self.assertThat(mapping_sql.list_id_mappings(), matchers.HasLength(initial_mappings + 1))
    self.assertEqual(public_id, PROVIDERS.id_mapping_api.get_public_id(local_entity))
    self.assertEqual(local_id, PROVIDERS.id_mapping_api.get_id_mapping(public_id)['local_id'])