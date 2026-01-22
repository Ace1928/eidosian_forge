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
def test_convert_policy_to_stdout(self):
    stdout = self._capture_stdout()
    self._test_convert_json_to_yaml_file(output_to_file=False)
    self.assertEqual(self.expected, stdout.getvalue())