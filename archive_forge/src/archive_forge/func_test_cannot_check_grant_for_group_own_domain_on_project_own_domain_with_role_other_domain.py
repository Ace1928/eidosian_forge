import uuid
import http.client
from oslo_serialization import jsonutils
from keystone.common.policies import grant as gp
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
def test_cannot_check_grant_for_group_own_domain_on_project_own_domain_with_role_other_domain(self):
    group_domain_id = self.domain_id
    project_domain_id = CONF.identity.default_domain_id
    role_domain_id = CONF.identity.default_domain_id
    role = PROVIDERS.role_api.create_role(uuid.uuid4().hex, unit.new_role_ref(domain_id=role_domain_id))
    group = PROVIDERS.identity_api.create_group(unit.new_group_ref(domain_id=group_domain_id))
    project = PROVIDERS.resource_api.create_project(uuid.uuid4().hex, unit.new_project_ref(domain_id=project_domain_id))
    with self.test_client() as c:
        c.get('/v3/projects/%s/groups/%s/roles/%s' % (project['id'], group['id'], role['id']), headers=self.headers, expected_status_code=http.client.FORBIDDEN)