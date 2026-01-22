import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def test_user_can_list_limits_within_domain(self):
    project_limit_id, domain_limit_id = _create_limits_and_dependencies(domain_id=self.domain_id)
    with self.test_client() as c:
        r = c.get('/v3/limits', headers=self.headers)
        result = []
        for limit in r.json['limits']:
            result.append(limit['id'])
        self.assertEqual(2, len(r.json['limits']))
        self.assertIn(project_limit_id, result)
        self.assertIn(domain_limit_id, result)