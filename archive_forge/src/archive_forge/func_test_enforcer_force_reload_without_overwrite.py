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
def test_enforcer_force_reload_without_overwrite(self, opts_registered=0):
    self.create_config_file(os.path.join('policy.d', 'a.conf'), POLICY_A_CONTENTS)
    self.create_config_file(os.path.join('policy.d', 'b.conf'), POLICY_B_CONTENTS)
    self.enforcer.set_rules({'test': _parser.parse_rule('role:test')}, use_conf=True)
    self.enforcer.set_rules({'default': _parser.parse_rule('role:fakeZ')}, overwrite=False, use_conf=True)
    self.enforcer.overwrite = False
    self.enforcer._is_directory_updated = lambda x, y: True
    self.assertTrue(self.enforcer.enforce('test', {}, {'roles': ['test']}))
    self.assertFalse(self.enforcer.enforce('default', {}, {'roles': ['fakeZ']}))
    self.assertIn('test', self.enforcer.rules)
    self.assertIn('default', self.enforcer.rules)
    self.assertIn('admin', self.enforcer.rules)
    loaded_rules = jsonutils.loads(str(self.enforcer.rules))
    self.assertEqual(3 + opts_registered, len(loaded_rules))
    self.assertIn('role:test', loaded_rules['test'])
    self.assertIn('role:fakeB', loaded_rules['default'])
    self.assertIn('is_admin:True', loaded_rules['admin'])