import datetime
import random
import uuid
import freezegun
import http.client
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.resource.backends import base as resource_base
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_check_user_does_not_have_system_role_without_assignment(self):
    system_role_id = self._create_new_role()
    member_url = '/system/users/%(user_id)s/roles/%(role_id)s' % {'user_id': self.user['id'], 'role_id': system_role_id}
    self.head(member_url, expected_status=http.client.NOT_FOUND)
    response = self.get('/role_assignments?scope.system=all&user.id=%(user_id)s' % {'user_id': self.user['id']})
    self.assertEqual(len(response.json_body['role_assignments']), 0)
    self.assertValidRoleAssignmentListResponse(response)