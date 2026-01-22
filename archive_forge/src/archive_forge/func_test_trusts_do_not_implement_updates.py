import datetime
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_trusts_do_not_implement_updates(self):
    with self.test_client() as c:
        token = self.get_scoped_token()
        ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=self.trustee_user_id, project_id=self.project_id, role_ids=[self.role_id])
        r = c.post('/v3/OS-TRUST/trusts', json={'trust': ref}, headers={'X-Auth-Token': token})
        trust_id = r.json['trust']['id']
        c.patch('/v3/OS-TRUST/trusts/%(trust_id)s' % {'trust_id': trust_id}, json={'trust': ref}, headers={'X-Auth-Token': token}, expected_status_code=http.client.METHOD_NOT_ALLOWED)
        c.put('/v3/OS-TRUST/trusts/%(trust_id)s' % {'trust_id': trust_id}, json={'trust': ref}, headers={'X-Auth-Token': token}, expected_status_code=http.client.METHOD_NOT_ALLOWED)