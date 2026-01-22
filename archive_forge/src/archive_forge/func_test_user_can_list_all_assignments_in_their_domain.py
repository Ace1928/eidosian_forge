import copy
import http.client
import uuid
from oslo_serialization import jsonutils
from keystone.common.policies import role_assignment as rp
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
def test_user_can_list_all_assignments_in_their_domain(self):
    self._setup_test_role_assignments()
    domain_assignments = self._setup_test_role_assignments_for_domain()
    self.expected.append({'user_id': domain_assignments['user_id'], 'domain_id': self.domain_id, 'role_id': domain_assignments['role_id']})
    self.expected.append({'user_id': domain_assignments['user_id'], 'project_id': domain_assignments['project_id'], 'role_id': domain_assignments['role_id']})
    self.expected.append({'group_id': domain_assignments['group_id'], 'domain_id': self.domain_id, 'role_id': domain_assignments['role_id']})
    self.expected.append({'group_id': domain_assignments['group_id'], 'project_id': domain_assignments['project_id'], 'role_id': domain_assignments['role_id']})
    with self.test_client() as c:
        r = c.get('/v3/role_assignments', headers=self.headers)
        self.assertEqual(len(self.expected), len(r.json['role_assignments']))
        actual = self._extract_role_assignments_from_response_body(r)
        for assignment in actual:
            self.assertIn(assignment, self.expected)