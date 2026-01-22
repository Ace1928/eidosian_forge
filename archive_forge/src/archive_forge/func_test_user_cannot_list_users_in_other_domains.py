import uuid
import http.client
from oslo_serialization import jsonutils
from keystone.common.policies import user as up
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
def test_user_cannot_list_users_in_other_domains(self):
    domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
    PROVIDERS.identity_api.create_user(unit.new_user_ref(domain_id=domain['id']))
    with self.test_client() as c:
        c.get('/v3/users?domain_id=%s' % domain['id'], headers=self.headers, expected_status_code=http.client.FORBIDDEN)