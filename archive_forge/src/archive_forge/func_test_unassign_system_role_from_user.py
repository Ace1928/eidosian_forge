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
def test_unassign_system_role_from_user(self):
    system_role_id = self._create_new_role()
    member_url = '/system/users/%(user_id)s/roles/%(role_id)s' % {'user_id': self.user['id'], 'role_id': system_role_id}
    self.put(member_url)
    self.head(member_url)
    response = self.get('/role_assignments?scope.system=all&user.id=%(user_id)s' % {'user_id': self.user['id']})
    self.assertEqual(len(response.json_body['role_assignments']), 1)
    self.assertValidRoleAssignmentListResponse(response)
    self.delete(member_url)
    collection_url = '/system/users/%(user_id)s/roles' % {'user_id': self.user['id']}
    response = self.get(collection_url)
    self.assertEqual(len(response.json_body['roles']), 0)
    response = self.get('/role_assignments?scope.system=all&user.id=%(user_id)s' % {'user_id': self.user['id']})
    self.assertValidRoleAssignmentListResponse(response, expected_length=0)