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
def test_admin_list_all_trusts(self):
    PROVIDERS.trust_api.create_trust(self.trust_id, **self.trust_data)
    with self.test_client() as c:
        r = c.get('/v3/OS-TRUST/trusts', headers=self.headers)
    self.assertEqual(1, len(r.json['trusts']))