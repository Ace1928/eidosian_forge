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
def test_delete_user_before_removing_system_assignments_succeeds(self):
    system_role = self._create_new_role()
    user = self._create_user()
    path = '/system/users/%(user_id)s/roles/%(role_id)s' % {'user_id': user['id'], 'role_id': system_role}
    self.put(path)
    response = self.get('/role_assignments')
    number_of_assignments = len(response.json_body['role_assignments'])
    path = '/users/%(user_id)s' % {'user_id': user['id']}
    self.delete(path)
    response = self.get('/role_assignments')
    self.assertValidRoleAssignmentListResponse(response, expected_length=number_of_assignments - 1)