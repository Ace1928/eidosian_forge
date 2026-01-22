import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def test_user_can_create_endpoint_group(self):
    create = {'endpoint_group': {'id': uuid.uuid4().hex, 'description': uuid.uuid4().hex, 'filters': {'interface': 'public'}, 'name': uuid.uuid4().hex}}
    with self.test_client() as c:
        c.post('/v3/OS-EP-FILTER/endpoint_groups', json=create, headers=self.headers)