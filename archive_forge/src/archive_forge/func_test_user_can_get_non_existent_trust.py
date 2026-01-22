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
def test_user_can_get_non_existent_trust(self):
    trust_id = uuid.uuid4().hex
    with self.test_client() as c:
        c.get('/v3/OS-TRUST/trusts/%s' % trust_id, headers=self.headers, expected_status_code=http.client.NOT_FOUND)