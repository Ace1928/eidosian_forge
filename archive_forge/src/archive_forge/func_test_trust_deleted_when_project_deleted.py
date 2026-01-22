import datetime
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_trust_deleted_when_project_deleted(self):
    ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=self.trustee_user_id, project_id=self.project_id, impersonation=False, role_ids=[self.role_id], allow_redelegation=True)
    resp = self.post('/OS-TRUST/trusts', body={'trust': ref})
    trust = self.assertValidTrustResponse(resp)
    r = self.get('/OS-TRUST/trusts')
    self.assertEqual(1, len(r.result['trusts']))
    self.delete('/projects/%(project_id)s' % {'project_id': trust['project_id']})
    self.assertRaises(exception.TrustNotFound, PROVIDERS.trust_api.get_trust, trust['id'])