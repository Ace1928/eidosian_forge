import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def test_user_can_get_a_limit(self):
    limit_id, _ = _create_limits_and_dependencies()
    with self.test_client() as c:
        r = c.get('/v3/limits/%s' % limit_id, headers=self.headers)
        self.assertEqual(limit_id, r.json['limit']['id'])