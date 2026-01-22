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
def test_enforcer_keep_use_conf_flag_after_reload_opts_registered(self):
    self.enforcer.register_default(policy.RuleDefault(name='admin', check_str='is_admin:False'))
    self.enforcer.register_default(policy.RuleDefault(name='owner', check_str='role:owner'))
    self.test_enforcer_keep_use_conf_flag_after_reload()
    self.assertIn('owner', self.enforcer.rules)
    self.assertEqual('role:owner', str(self.enforcer.rules['owner']))
    self.assertEqual('is_admin:True', str(self.enforcer.rules['admin']))