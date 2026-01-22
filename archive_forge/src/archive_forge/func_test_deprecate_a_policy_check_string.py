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
@mock.patch('warnings.warn', new=mock.Mock())
def test_deprecate_a_policy_check_string(self):
    deprecated_rule = policy.DeprecatedRule(name='foo:create_bar', check_str='role:fizz', deprecated_reason='"role:bang" is a better default', deprecated_since='N')
    rule_list = [policy.DocumentedRuleDefault(name='foo:create_bar', check_str='role:bang', description='Create a bar.', operations=[{'path': '/v1/bars', 'method': 'POST'}], deprecated_rule=deprecated_rule)]
    enforcer = policy.Enforcer(self.conf)
    enforcer.register_defaults(rule_list)
    expected_msg = 'Policy "foo:create_bar":"role:fizz" was deprecated in N in favor of "foo:create_bar":"role:bang". Reason: "role:bang" is a better default. Either ensure your deployment is ready for the new default or copy/paste the deprecated policy into your policy file and maintain it manually.'
    with mock.patch('warnings.warn') as mock_warn:
        enforcer.load_rules()
        mock_warn.assert_called_once_with(expected_msg)
    self.assertTrue(enforcer.enforce('foo:create_bar', {}, {'roles': ['bang']}))
    self.assertTrue(enforcer.enforce('foo:create_bar', {}, {'roles': ['fizz']}))
    self.assertFalse(enforcer.enforce('foo:create_bar', {}, {'roles': ['baz']}))