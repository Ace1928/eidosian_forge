import os
from unittest import mock
import yaml
import fixtures
from oslo_config import cfg
from oslo_context import context
from oslo_serialization import jsonutils
from oslotest import base as test_base
from oslo_policy import _cache_handler
from oslo_policy import _checks
from oslo_policy import _parser
from oslo_policy import policy
from oslo_policy.tests import base
@mock.patch('oslo_policy.policy.Enforcer.check_rules')
def test_load_rules_twice(self, mock_check_rules):
    self.enforcer.load_rules()
    self.enforcer.load_rules()
    self.assertEqual(1, mock_check_rules.call_count)