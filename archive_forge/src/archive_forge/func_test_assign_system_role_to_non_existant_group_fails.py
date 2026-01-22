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
def test_assign_system_role_to_non_existant_group_fails(self):
    system_role_id = self._create_new_role()
    group_id = uuid.uuid4().hex
    member_url = '/system/groups/%(group_id)s/roles/%(role_id)s' % {'group_id': group_id, 'role_id': system_role_id}
    self.put(member_url, expected_status=http.client.NOT_FOUND)