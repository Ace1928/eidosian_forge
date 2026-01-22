import operator
from unittest import mock
import warnings
from oslo_config import cfg
import stevedore
import testtools
import yaml
from oslo_policy import generator
from oslo_policy import policy
from oslo_policy.tests import base
from oslo_serialization import jsonutils
def test_converted_yaml_is_loadable(self):
    self._test_convert_json_to_yaml_file()
    enforcer = policy.Enforcer(self.conf, policy_file=self.output_file_path)
    enforcer.load_rules()
    for rule in ['rule2_name', 'deprecated_rule1_name']:
        self.assertIn(rule, enforcer.rules)