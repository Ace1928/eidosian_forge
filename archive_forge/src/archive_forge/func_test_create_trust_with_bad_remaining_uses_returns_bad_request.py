import datetime
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_create_trust_with_bad_remaining_uses_returns_bad_request(self):
    for value in [-1, 0, 'a bad value', 7.2]:
        ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=self.trustee_user_id, project_id=self.project_id, remaining_uses=value, role_ids=[self.role_id])
        self.post('/OS-TRUST/trusts', body={'trust': ref}, expected_status=http.client.BAD_REQUEST)