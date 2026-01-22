import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def test_user_cannot_create_consumer(self):
    with self.test_client() as c:
        c.post('/v3/OS-OAUTH1/consumers', json={'consumer': {}}, expected_status_code=http.client.FORBIDDEN, headers=self.headers)