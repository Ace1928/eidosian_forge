import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def test_can_get_security_compliance_config_with_user_from_other_domain(self):
    domain = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(domain['id'], domain)
    password_regex = uuid.uuid4().hex
    password_regex_description = uuid.uuid4().hex
    group = 'security_compliance'
    self.config_fixture.config(group=group, password_regex=password_regex)
    self.config_fixture.config(group=group, password_regex_description=password_regex_description)
    with self.test_client() as c:
        c.get('/v3/domains/%s/config/security_compliance' % CONF.identity.default_domain_id, headers=self.headers)