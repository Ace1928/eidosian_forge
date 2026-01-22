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
def test_deprecate_an_empty_policy_check_string(self):
    deprecated_rule = policy.DeprecatedRule(name='foo:create_bar', check_str='', deprecated_reason='because of reasons', deprecated_since='N')
    rule_list = [policy.DocumentedRuleDefault(name='foo:create_bar', check_str='role:bang', description='Create a bar.', operations=[{'path': '/v1/bars', 'method': 'POST'}], deprecated_rule=deprecated_rule)]
    enforcer = policy.Enforcer(self.conf)
    enforcer.register_defaults(rule_list)
    with mock.patch('warnings.warn') as mock_warn:
        enforcer.load_rules()
        mock_warn.assert_called_once()
    enforcer.enforce('foo:create_bar', {}, {'roles': ['bang']}, do_raise=True)
    enforcer.enforce('foo:create_bar', {}, {'roles': ['fizz']}, do_raise=True)