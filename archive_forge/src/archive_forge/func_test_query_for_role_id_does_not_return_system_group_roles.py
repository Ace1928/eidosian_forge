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
def test_query_for_role_id_does_not_return_system_group_roles(self):
    system_role_id = self._create_new_role()
    group = self._create_group()
    member_url = '/system/groups/%(group_id)s/roles/%(role_id)s' % {'group_id': group['id'], 'role_id': system_role_id}
    self.put(member_url)
    member_url = '/projects/%(project_id)s/groups/%(group_id)s/roles/%(role_id)s' % {'project_id': self.project_id, 'group_id': group['id'], 'role_id': self.role_id}
    self.put(member_url)
    path = '/role_assignments?role.id=%(role_id)s&group.id=%(group_id)s' % {'role_id': self.role_id, 'group_id': group['id']}
    response = self.get(path)
    self.assertValidRoleAssignmentListResponse(response, expected_length=1)