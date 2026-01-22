import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def test_user_can_get_config_by_group_invalid_domain(self):
    domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
    PROVIDERS.domain_config_api.create_config(domain['id'], unit.new_domain_config_ref())
    invalid_domain_id = uuid.uuid4().hex
    with self.test_client() as c:
        c.get('/v3/domains/%s/config/ldap' % invalid_domain_id, headers=self.headers, expected_status_code=http.client.NOT_FOUND)