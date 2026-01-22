import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def test_user_cannot_update_limits_for_projects_outside_domain(self):
    project_limit_id, _ = _create_limits_and_dependencies()
    update = {'limit': {'resource_limit': 1}}
    with self.test_client() as c:
        c.patch('/v3/limits/%s' % project_limit_id, headers=self.headers, json=update, expected_status_code=http.client.FORBIDDEN)