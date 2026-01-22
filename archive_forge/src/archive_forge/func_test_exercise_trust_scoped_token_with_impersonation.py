import datetime
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_exercise_trust_scoped_token_with_impersonation(self):
    ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=self.trustee_user_id, project_id=self.project_id, impersonation=True, expires=dict(minutes=1), role_ids=[self.role_id])
    resp = self.post('/OS-TRUST/trusts', body={'trust': ref})
    trust = self.assertValidTrustResponse(resp)
    auth_data = self.build_authentication_request(user_id=self.trustee_user['id'], password=self.trustee_user['password'], trust_id=trust['id'])
    resp = self.v3_create_token(auth_data)
    resp_body = resp.json_body['token']
    self.assertValidProjectScopedTokenResponse(resp, self.user)
    self.assertEqual(self.user['id'], resp_body['user']['id'])
    self.assertEqual(self.user['name'], resp_body['user']['name'])
    self.assertEqual(self.domain['id'], resp_body['user']['domain']['id'])
    self.assertEqual(self.domain['name'], resp_body['user']['domain']['name'])
    self.assertEqual(self.project['id'], resp_body['project']['id'])
    self.assertEqual(self.project['name'], resp_body['project']['name'])