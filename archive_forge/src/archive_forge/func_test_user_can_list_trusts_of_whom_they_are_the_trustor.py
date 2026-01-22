import uuid
import http.client
from oslo_serialization import jsonutils
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
def test_user_can_list_trusts_of_whom_they_are_the_trustor(self):
    PROVIDERS.trust_api.create_trust(self.trust_id, **self.trust_data)
    with self.test_client() as c:
        r = c.get('/v3/OS-TRUST/trusts?trustor_user_id=%s' % self.trustor_user_id, headers=self.trustor_headers)
    self.assertEqual(1, len(r.json['trusts']))
    self.assertEqual(self.trust_id, r.json['trusts'][0]['id'])