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
def test_enforcer_with_default_rule(self):
    rules_json = jsonutils.dumps({'deny_stack_user': 'not role:stack_user', 'cloudwatch:PutMetricData': ''})
    rules = policy.Rules.load(rules_json)
    default_rule = _checks.TrueCheck()
    enforcer = policy.Enforcer(self.conf, default_rule=default_rule)
    enforcer.set_rules(rules)
    action = 'cloudwatch:PutMetricData'
    creds = {'roles': ''}
    self.assertTrue(enforcer.enforce(action, {}, creds))