import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def test_user_can_list_mappings(self):
    mapping = unit.new_mapping_ref()
    mapping = PROVIDERS.federation_api.create_mapping(mapping['id'], mapping)
    with self.test_client() as c:
        r = c.get('/v3/OS-FEDERATION/mappings', headers=self.headers)
        self.assertEqual(1, len(r.json['mappings']))
        self.assertEqual(mapping['id'], r.json['mappings'][0]['id'])