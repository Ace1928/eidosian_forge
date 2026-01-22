import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
def test_user_cannot_get_own_non_existent_access_rule_not_found(self):
    with self.test_client() as c:
        c.get('/v3/users/%s/access_rules/%s' % (self.user_id, uuid.uuid4().hex), headers=self.headers, expected_status_code=http.client.NOT_FOUND)