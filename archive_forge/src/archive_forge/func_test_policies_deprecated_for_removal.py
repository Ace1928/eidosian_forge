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
def test_policies_deprecated_for_removal(self):
    rule = policy.RuleDefault(name='foo:post_bar', check_str='role:fizz', description='Create a bar.', deprecated_for_removal=True, deprecated_reason='This policy is not used anymore', deprecated_since='N')
    opts = {'rules': [rule]}
    extensions = []
    for name, opts in opts.items():
        ext = stevedore.extension.Extension(name=name, entry_point=None, plugin=None, obj=opts)
        extensions.append(ext)
    test_mgr = stevedore.named.NamedExtensionManager.make_test_instance(extensions=extensions, namespace=['rules'])
    expected = '# DEPRECATED\n# "foo:post_bar" has been deprecated since N.\n# This policy is not used anymore\n# Create a bar.\n#"foo:post_bar": "role:fizz"\n\n'
    stdout = self._capture_stdout()
    with mock.patch('stevedore.named.NamedExtensionManager', return_value=test_mgr) as mock_ext_mgr:
        generator._generate_sample(['rules'], output_file=None)
        mock_ext_mgr.assert_called_once_with('oslo.policy.policies', names=['rules'], on_load_failure_callback=generator.on_load_failure_callback, invoke_on_load=True)
    self.assertEqual(expected, stdout.getvalue())