import uuid
import http.client
from oslo_serialization import jsonutils
from keystone.common.policies import base
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
def test_user_can_grant_system_assignments(self):
    user = PROVIDERS.identity_api.create_user(unit.new_user_ref(CONF.identity.default_domain_id))
    with self.test_client() as c:
        c.put('/v3/system/users/%s/roles/%s' % (user['id'], self.bootstrapper.member_role_id), headers=self.headers)