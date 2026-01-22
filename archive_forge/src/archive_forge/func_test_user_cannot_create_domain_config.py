import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def test_user_cannot_create_domain_config(self):
    domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
    with self.test_client() as c:
        c.put('/v3/domains/%s/config' % domain['id'], json={'config': unit.new_domain_config_ref()}, headers=self.headers, expected_status_code=http.client.FORBIDDEN)