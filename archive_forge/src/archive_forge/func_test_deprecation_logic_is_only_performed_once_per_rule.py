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
def test_deprecation_logic_is_only_performed_once_per_rule(self):
    deprecated_rule = policy.DeprecatedRule(name='foo:create_bar', check_str='role:fizz')
    rule = policy.DocumentedRuleDefault(name='foo:create_bar', check_str='role:bang', description='Create a bar.', operations=[{'path': '/v1/bars', 'method': 'POST'}], deprecated_rule=deprecated_rule, deprecated_reason='"role:bang" is a better default', deprecated_since='N')
    check = rule.check
    enforcer = policy.Enforcer(self.conf)
    enforcer.register_defaults([rule])
    self.assertEqual({}, enforcer.rules)
    enforcer.load_rules()
    expected_check = policy.OrCheck([_parser.parse_rule(cs) for cs in [rule.check_str, deprecated_rule.check_str]])
    self.assertIn('foo:create_bar', enforcer.rules)
    self.assertEqual(str(enforcer.rules['foo:create_bar']), str(expected_check))
    self.assertEqual(check, rule.check)
    enforcer.rules['foo:create_bar'] = 'foo:bar'
    enforcer.load_rules()
    self.assertEqual('foo:bar', enforcer.rules['foo:create_bar'])