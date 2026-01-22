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
def test_templatized_enforcement(self):
    target_mine = {'project_id': 'fake'}
    target_not_mine = {'project_id': 'another'}
    credentials = {'project_id': 'fake', 'roles': []}
    action = 'example:my_file'
    policy.enforce(credentials, action, target_mine)
    self.assertRaises(exception.ForbiddenAction, policy.enforce, credentials, action, target_not_mine)