import uuid
from testtools import matchers
from keystone.common import provider_api
from keystone.common import sql
from keystone.identity.mapping_backends import mapping
from keystone.tests import unit
from keystone.tests.unit import identity_mapping as mapping_sql
from keystone.tests.unit import test_backend_sql
def test_get_domain_mapping_list_by_group_entity_type(self):
    local_entities = self._prepare_domain_mappings_for_list()
    with sql.session_for_read():
        domain_b_mappings_group = PROVIDERS.id_mapping_api.get_domain_mapping_list(self.domainB['id'], entity_type=mapping.EntityType.GROUP)
        domain_b_mappings_group = domain_b_mappings_group.first().to_dict()
    self.assertCountEqual(local_entities[2], domain_b_mappings_group)