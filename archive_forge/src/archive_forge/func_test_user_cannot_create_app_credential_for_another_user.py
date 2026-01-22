import datetime
import uuid
import http.client
from oslo_serialization import jsonutils
from keystone.common.policies import base as base_policy
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
def test_user_cannot_create_app_credential_for_another_user(self):
    another_user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
    another_user_id = PROVIDERS.identity_api.create_user(another_user)['id']
    app_cred_body = {'application_credential': unit.new_application_credential_ref(roles=[{'id': self.bootstrapper.member_role_id}])}
    with self.test_client() as c:
        c.post('/v3/users/%s/application_credentials' % another_user_id, json=app_cred_body, expected_status_code=http.client.FORBIDDEN, headers=self.headers)