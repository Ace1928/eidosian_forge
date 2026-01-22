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
def test_assign_system_role_to_group(self):
    system_role_id = self._create_new_role()
    group = self._create_group()
    member_url = '/system/groups/%(group_id)s/roles/%(role_id)s' % {'group_id': group['id'], 'role_id': system_role_id}
    self.put(member_url)
    self.head(member_url)
    collection_url = '/system/groups/%(group_id)s/roles' % {'group_id': group['id']}
    roles = self.get(collection_url).json_body['roles']
    self.assertEqual(len(roles), 1)
    self.assertEqual(roles[0]['id'], system_role_id)
    self.head(collection_url, expected_status=http.client.OK)
    response = self.get('/role_assignments?scope.system=all&group.id=%(group_id)s' % {'group_id': group['id']})
    self.assertValidRoleAssignmentListResponse(response, expected_length=1)
    self.assertEqual(response.json_body['role_assignments'][0]['role']['id'], system_role_id)