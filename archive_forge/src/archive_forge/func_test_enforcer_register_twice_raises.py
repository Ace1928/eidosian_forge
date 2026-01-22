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
def test_enforcer_register_twice_raises(self):
    self.enforcer.register_default(policy.RuleDefault(name='owner', check_str='role:owner'))
    self.assertRaises(policy.DuplicatePolicyError, self.enforcer.register_default, policy.RuleDefault(name='owner', check_str='role:owner'))