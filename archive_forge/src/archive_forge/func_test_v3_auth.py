import copy
from oslo_serialization import jsonutils
from oslo_utils import uuidutils
from requests_mock.contrib import fixture
import testtools
import barbicanclient.barbican
def test_v3_auth(self):
    self.responses.get(V3_URL, text=V3_VERSION_ENTRY)
    id, v3_token = generate_v3_project_scoped_token()
    self.responses.post('{0}/auth/tokens'.format(V3_URL), json=v3_token, headers={'X-Subject-Token': '1234'})
    self._delete_secret('v3')