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
def test_trustor_can_create_trust(self):
    json = {'trust': self.trust_data['trust']}
    json['trust']['roles'] = self.trust_data['roles']
    with self.test_client() as c:
        c.post('/v3/OS-TRUST/trusts', json=json, headers=self.trustor_headers)