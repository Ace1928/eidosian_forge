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
def test_deprecated_policies_are_aliased_to_new_names(self):
    deprecated_rule = policy.DeprecatedRule(name='foo:post_bar', check_str='role:fizz', deprecated_reason='foo:post_bar is being removed in favor of foo:create_bar', deprecated_since='N')
    new_rule = policy.RuleDefault(name='foo:create_bar', check_str='role:fizz', description='Create a bar.', deprecated_rule=deprecated_rule)
    opts = {'rules': [new_rule]}
    extensions = []
    for name, opts in opts.items():
        ext = stevedore.extension.Extension(name=name, entry_point=None, plugin=None, obj=opts)
        extensions.append(ext)
    test_mgr = stevedore.named.NamedExtensionManager.make_test_instance(extensions=extensions, namespace=['rules'])
    expected = '# Create a bar.\n#"foo:create_bar": "role:fizz"\n\n# DEPRECATED\n# "foo:post_bar":"role:fizz" has been deprecated since N in favor of\n# "foo:create_bar":"role:fizz".\n# foo:post_bar is being removed in favor of foo:create_bar\n# WARNING: A rule name change has been identified.\n#          This may be an artifact of new rules being\n#          included which require legacy fallback\n#          rules to ensure proper policy behavior.\n#          Alternatively, this may just be an alias.\n#          Please evaluate on a case by case basis\n#          keeping in mind the format for aliased\n#          rules is:\n#          "old_rule_name": "new_rule_name".\n# "foo:post_bar": "rule:foo:create_bar"\n\n'
    stdout = self._capture_stdout()
    with mock.patch('stevedore.named.NamedExtensionManager', return_value=test_mgr) as mock_ext_mgr:
        generator._generate_sample(['rules'], output_file=None)
        mock_ext_mgr.assert_called_once_with('oslo.policy.policies', names=['rules'], on_load_failure_callback=generator.on_load_failure_callback, invoke_on_load=True)
    self.assertEqual(expected, stdout.getvalue())