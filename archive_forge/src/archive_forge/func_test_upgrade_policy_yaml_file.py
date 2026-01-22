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
def test_upgrade_policy_yaml_file(self):
    test_mgr = stevedore.named.NamedExtensionManager.make_test_instance(extensions=self.extensions, namespace='test_upgrade')
    with mock.patch('stevedore.named.NamedExtensionManager', return_value=test_mgr):
        testargs = ['olsopolicy-policy-upgrade', '--policy', self.get_config_file_fullname('policy.json'), '--namespace', 'test_upgrade', '--output-file', self.get_config_file_fullname('new_policy.yaml'), '--format', 'yaml']
        with mock.patch('sys.argv', testargs):
            generator.upgrade_policy(conf=self.local_conf)
            new_file = self.get_config_file_fullname('new_policy.yaml')
            with open(new_file, 'r') as fh:
                new_policy = yaml.safe_load(fh)
            self.assertIsNotNone(new_policy.get('new_policy_name'))
            self.assertIsNone(new_policy.get('deprecated_name'))