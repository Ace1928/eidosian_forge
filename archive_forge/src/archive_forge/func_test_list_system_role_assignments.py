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
def test_list_system_role_assignments(self):
    user_system_role_id = self._create_new_role()
    user_domain_role_id = self._create_new_role()
    user_project_role_id = self._create_new_role()
    group_system_role_id = self._create_new_role()
    group_domain_role_id = self._create_new_role()
    group_project_role_id = self._create_new_role()
    user = self._create_user()
    url = '/system/users/%s/roles/%s' % (user['id'], user_system_role_id)
    self.put(url)
    url = '/domains/%s/users/%s/roles/%s' % (self.domain_id, user['id'], user_domain_role_id)
    self.put(url)
    url = '/projects/%s/users/%s/roles/%s' % (self.project_id, user['id'], user_project_role_id)
    self.put(url)
    group = self._create_group()
    url = '/system/groups/%s/roles/%s' % (group['id'], group_system_role_id)
    self.put(url)
    url = '/domains/%s/groups/%s/roles/%s' % (self.domain_id, group['id'], group_domain_role_id)
    self.put(url)
    url = '/projects/%s/groups/%s/roles/%s' % (self.project_id, group['id'], group_project_role_id)
    self.put(url)
    response = self.get('/role_assignments?scope.system=all')
    self.assertValidRoleAssignmentListResponse(response, expected_length=2)
    for assignment in response.json_body['role_assignments']:
        self.assertTrue(assignment['scope']['system']['all'])
        if assignment.get('user'):
            self.assertEqual(user_system_role_id, assignment['role']['id'])
        if assignment.get('group'):
            self.assertEqual(group_system_role_id, assignment['role']['id'])
    url = '/role_assignments?scope.system=all&user.id=%s' % user['id']
    response = self.get(url)
    self.assertValidRoleAssignmentListResponse(response, expected_length=1)
    self.assertEqual(user_system_role_id, response.json_body['role_assignments'][0]['role']['id'])
    url = '/role_assignments?scope.system=all&group.id=%s' % group['id']
    response = self.get(url)
    self.assertValidRoleAssignmentListResponse(response, expected_length=1)
    self.assertEqual(group_system_role_id, response.json_body['role_assignments'][0]['role']['id'])
    url = '/role_assignments?user.id=%s' % user['id']
    response = self.get(url)
    self.assertValidRoleAssignmentListResponse(response, expected_length=3)
    for assignment in response.json_body['role_assignments']:
        if 'system' in assignment['scope']:
            self.assertEqual(user_system_role_id, assignment['role']['id'])
        if 'domain' in assignment['scope']:
            self.assertEqual(user_domain_role_id, assignment['role']['id'])
        if 'project' in assignment['scope']:
            self.assertEqual(user_project_role_id, assignment['role']['id'])
    url = '/role_assignments?group.id=%s' % group['id']
    response = self.get(url)
    self.assertValidRoleAssignmentListResponse(response, expected_length=3)
    for assignment in response.json_body['role_assignments']:
        if 'system' in assignment['scope']:
            self.assertEqual(group_system_role_id, assignment['role']['id'])
        if 'domain' in assignment['scope']:
            self.assertEqual(group_domain_role_id, assignment['role']['id'])
        if 'project' in assignment['scope']:
            self.assertEqual(group_project_role_id, assignment['role']['id'])