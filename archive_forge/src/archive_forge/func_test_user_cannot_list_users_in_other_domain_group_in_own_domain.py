import uuid
import http.client
from oslo_serialization import jsonutils
from keystone.common.policies import group as gp
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
def test_user_cannot_list_users_in_other_domain_group_in_own_domain(self):
    domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
    user1 = PROVIDERS.identity_api.create_user(unit.new_user_ref(domain_id=domain['id']))
    user2 = PROVIDERS.identity_api.create_user(unit.new_user_ref(domain_id=self.domain_id))
    group = PROVIDERS.identity_api.create_group(unit.new_group_ref(domain_id=self.domain_id))
    PROVIDERS.identity_api.add_user_to_group(user1['id'], group['id'])
    PROVIDERS.identity_api.add_user_to_group(user2['id'], group['id'])
    with self.test_client() as c:
        r = c.get('/v3/groups/%s/users' % group['id'], headers=self.headers)
        self.assertEqual(1, len(r.json['users']))
        self.assertEqual(user2['id'], r.json['users'][0]['id'])