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
def test_enforcer_force_reload_false(self):
    self.enforcer.set_rules({'test': 'test'})
    self.enforcer.load_rules(force_reload=False)
    self.assertIn('test', self.enforcer.rules)
    self.assertNotIn('default', self.enforcer.rules)
    self.assertNotIn('admin', self.enforcer.rules)