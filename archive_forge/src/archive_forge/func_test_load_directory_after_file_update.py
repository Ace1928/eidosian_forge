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
def test_load_directory_after_file_update(self):
    self.create_config_file(os.path.join('policy.d', 'a.conf'), POLICY_A_CONTENTS)
    self.enforcer.load_rules(True)
    self.assertIsNotNone(self.enforcer.rules)
    loaded_rules = jsonutils.loads(str(self.enforcer.rules))
    self.assertEqual('role:fakeA', loaded_rules['default'])
    self.assertEqual('is_admin:True', loaded_rules['admin'])
    new_policy_json_contents = jsonutils.dumps({'default': 'rule:admin', 'admin': 'is_admin:True', 'foo': 'rule:bar'})
    self.create_config_file('policy.json', new_policy_json_contents)
    policy_file_path = self.get_config_file_fullname('policy.json')
    stinfo = os.stat(policy_file_path)
    os.utime(policy_file_path, (stinfo.st_atime + 42, stinfo.st_mtime + 42))
    self.enforcer.load_rules()
    self.assertIsNotNone(self.enforcer.rules)
    loaded_rules = jsonutils.loads(str(self.enforcer.rules))
    self.assertEqual('role:fakeA', loaded_rules['default'])
    self.assertEqual('is_admin:True', loaded_rules['admin'])
    self.assertEqual('rule:bar', loaded_rules['foo'])