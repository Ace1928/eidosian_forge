import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def test_user_cannot_update_protocols(self):
    protocol, mapping, identity_provider = self._create_protocol_and_deps()
    new_mapping = PROVIDERS.federation_api.create_mapping(uuid.uuid4().hex, unit.new_mapping_ref())
    update = {'protocol': {'mapping_id': new_mapping['id']}}
    with self.test_client() as c:
        path = '/v3/OS-FEDERATION/identity_providers/%s/protocols/%s' % (identity_provider['id'], protocol['id'])
        c.patch(path, json=update, headers=self.headers, expected_status_code=http.client.FORBIDDEN)