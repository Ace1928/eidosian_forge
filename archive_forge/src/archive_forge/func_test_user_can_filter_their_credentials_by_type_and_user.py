import uuid
import http.client
from oslo_serialization import jsonutils
from keystone.common.policies import base as bp
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
def test_user_can_filter_their_credentials_by_type_and_user(self):
    with self.test_client() as c:
        credential_type = uuid.uuid4().hex
        create = {'credential': {'blob': uuid.uuid4().hex, 'type': credential_type, 'user_id': self.user_id}}
        r = c.post('/v3/credentials', json=create, headers=self.headers)
        expected_credential_id = r.json['credential']['id']
        create = {'credential': {'blob': uuid.uuid4().hex, 'type': uuid.uuid4().hex, 'user_id': self.user_id}}
        r = c.post('/v3/credentials', json=create, headers=self.headers)
        path = '/v3/credentials?type=%s' % credential_type
        r = c.get(path, headers=self.headers)
        self.assertEqual(expected_credential_id, r.json['credentials'][0]['id'])
        path = '/v3/credentials?user=%s' % self.user_id
        r = c.get(path, headers=self.headers)
        self.assertEqual(expected_credential_id, r.json['credentials'][0]['id'])