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
def test_convert_json_to_yaml_file(self):
    converted_policy_data = self._test_convert_json_to_yaml_file()
    self.assertTrue(self._is_yaml(converted_policy_data))
    self.assertEqual(self.expected, converted_policy_data)