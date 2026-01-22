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
def test_rule_with_check(self):
    rules_json = jsonutils.dumps({'deny_stack_user': 'not role:stack_user', 'cloudwatch:PutMetricData': ''})
    rules = policy.Rules.load(rules_json)
    self.enforcer.set_rules(rules)
    action = 'cloudwatch:PutMetricData'
    creds = {'roles': ''}
    self.assertTrue(self.enforcer.enforce(action, {}, creds))