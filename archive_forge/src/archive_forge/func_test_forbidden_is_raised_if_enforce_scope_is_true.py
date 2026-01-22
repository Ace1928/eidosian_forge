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
def test_forbidden_is_raised_if_enforce_scope_is_true(self):
    self.config_fixture.config(group='oslo_policy', enforce_scope=True)
    self.assertRaises(exception.ForbiddenAction, policy.enforce, self.credentials, self.action, self.target)