from collections import abc
from unittest import mock
import hashlib
import os.path
import oslo_config.cfg
from oslo_policy import policy as common_policy
import glance.api.policy
from glance.common import exception
import glance.context
from glance.policies import base as base_policy
from glance.tests.unit import base
def test_project_member_check_string(self):
    expected = 'role:member and project_id:%(project_id)s'
    self.assertEqual(expected, base_policy.PROJECT_MEMBER)