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
def test_opts_registered(self):
    self.enforcer.register_default(policy.RuleDefault(name='admin', check_str='is_admin:False'))
    self.enforcer.register_default(policy.RuleDefault(name='owner', check_str='role:owner'))
    self.enforcer.load_rules(True)
    self.assertEqual({}, self.enforcer.file_rules)
    self.assertEqual('role:owner', str(self.enforcer.rules['owner']))
    self.assertEqual('is_admin:False', str(self.enforcer.rules['admin']))