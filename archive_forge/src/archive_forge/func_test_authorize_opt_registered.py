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
def test_authorize_opt_registered(self):
    self.enforcer.register_default(policy.RuleDefault(name='test', check_str='role:test'))
    self.assertTrue(self.enforcer.authorize('test', {}, {'roles': ['test']}))