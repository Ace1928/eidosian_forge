import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def test_user_cannot_list_limits_without_project_role_assignment(self):
    _create_limits_and_dependencies()
    with self.test_client() as c:
        r = c.get('/v3/limits', headers=self.headers)
        self.assertEqual(0, len(r.json['limits']))