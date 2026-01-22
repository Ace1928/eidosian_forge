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
def test_enforce_new_defaults_no_old_check_string(self):
    self.conf.set_override('enforce_new_defaults', True, group='oslo_policy')
    deprecated_rule = policy.DeprecatedRule(name='foo:create_bar', check_str='role:fizz', deprecated_reason='"role:bang" is a better default', deprecated_since='N')
    rule_list = [policy.DocumentedRuleDefault(name='foo:create_bar', check_str='role:bang', description='Create a bar.', operations=[{'path': '/v1/bars', 'method': 'POST'}], deprecated_rule=deprecated_rule)]
    enforcer = policy.Enforcer(self.conf)
    enforcer.register_defaults(rule_list)
    with mock.patch('warnings.warn') as mock_warn:
        enforcer.load_rules()
        mock_warn.assert_not_called()
    self.assertTrue(enforcer.enforce('foo:create_bar', {}, {'roles': ['bang']}))
    self.assertFalse(enforcer.enforce('foo:create_bar', {}, {'roles': ['fizz']}))
    self.assertFalse(enforcer.enforce('foo:create_bar', {}, {'roles': ['baz']}))