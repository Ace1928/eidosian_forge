import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
def test_user_cannot_list_access_rules_for_other_users(self):
    user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
    user = PROVIDERS.identity_api.create_user(user)
    project = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
    project = PROVIDERS.resource_api.create_project(project['id'], project)
    PROVIDERS.assignment_api.create_grant(self.bootstrapper.member_role_id, user_id=user['id'], project_id=project['id'])
    app_cred = {'id': uuid.uuid4().hex, 'name': uuid.uuid4().hex, 'user_id': user['id'], 'project_id': project['id'], 'secret': uuid.uuid4().hex, 'access_rules': [{'id': uuid.uuid4().hex, 'service': uuid.uuid4().hex, 'path': uuid.uuid4().hex, 'method': uuid.uuid4().hex[16:]}]}
    PROVIDERS.application_credential_api.create_application_credential(app_cred)
    with self.test_client() as c:
        path = '/v3/users/%s/access_rules' % user['id']
        c.get(path, headers=self.headers, expected_status_code=http.client.FORBIDDEN)