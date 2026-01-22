import os
import subprocess
from unittest import mock
import uuid
from oslo_policy import policy as common_policy
from keystone.common import policies
from keystone.common.rbac_enforcer import policy
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
def test_policies_loads(self):
    action = 'identity:list_projects'
    target = {'user_id': uuid.uuid4().hex, 'user.domain_id': uuid.uuid4().hex, 'group.domain_id': uuid.uuid4().hex, 'project.domain_id': uuid.uuid4().hex, 'project_id': uuid.uuid4().hex, 'domain_id': uuid.uuid4().hex}
    credentials = {'username': uuid.uuid4().hex, 'token': uuid.uuid4().hex, 'project_name': None, 'user_id': uuid.uuid4().hex, 'roles': [u'admin'], 'is_admin': True, 'is_admin_project': True, 'project_id': None, 'domain_id': uuid.uuid4().hex}
    result = policy._ENFORCER._enforcer.enforce(action, target, credentials)
    self.assertTrue(result)