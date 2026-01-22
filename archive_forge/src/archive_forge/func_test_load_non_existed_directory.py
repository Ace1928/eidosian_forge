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
def test_load_non_existed_directory(self):
    self.create_config_file(os.path.join('policy.d', 'a.conf'), POLICY_A_CONTENTS)
    self.conf.set_override('policy_dirs', ['policy.d', 'policy.x.d'], group='oslo_policy')
    self.enforcer.load_rules(True)
    self.assertIsNotNone(self.enforcer.rules)
    self.assertIn('default', self.enforcer.rules)
    self.assertIn('admin', self.enforcer.rules)