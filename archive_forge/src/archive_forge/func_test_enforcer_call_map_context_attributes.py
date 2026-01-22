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
@mock.patch.object(policy.Enforcer, '_map_context_attributes_into_creds')
def test_enforcer_call_map_context_attributes(self, map_mock):
    map_mock.return_value = {}
    rule = policy.RuleDefault(name='fake_rule', check_str='role:test')
    self.enforcer.register_default(rule)
    request_context = context.RequestContext()
    target_dict = {}
    self.enforcer.enforce('fake_rule', target_dict, request_context)
    map_mock.assert_called_once_with(request_context)