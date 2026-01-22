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
def test_deprecated_policy_should_not_include_deprecated_meta(self):
    deprecated_rule = policy.DeprecatedRule(name='foo:bar', check_str='rule:baz')
    with mock.patch('warnings.warn') as mock_warn:
        policy.DocumentedRuleDefault(name='foo:bar', check_str='rule:baz', description='Create a foo.', operations=[{'path': '/v1/foos/', 'method': 'POST'}], deprecated_rule=deprecated_rule, deprecated_reason='Some reason.')
        mock_warn.assert_called_once()