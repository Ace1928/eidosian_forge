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
def test_deprecated_policy_for_removal_must_include_deprecated_meta(self):
    self.assertRaises(ValueError, policy.DocumentedRuleDefault, name='foo:bar', check_str='rule:baz', description='Create a foo.', operations=[{'path': '/v1/foos/', 'method': 'POST'}], deprecated_for_removal=True, deprecated_reason='Some reason.')