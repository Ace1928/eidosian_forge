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
def test_user_cannot_list_trusts_for_other_trustor_overridden(self):
    self._override_policy_old_defaults()
    PROVIDERS.trust_api.create_trust(self.trust_id, **self.trust_data)
    with self.test_client() as c:
        c.get('/v3/OS-TRUST/trusts?trustor_user_id=%s' % self.trustor_user_id, headers=self.other_headers, expected_status_code=http.client.FORBIDDEN)