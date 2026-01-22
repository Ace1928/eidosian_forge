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
def test_str_true(self):
    exemplar = jsonutils.dumps({'admin_or_owner': ''}, indent=4)
    rules = policy.Rules(dict(admin_or_owner=_checks.TrueCheck()))
    self.assertEqual(exemplar, str(rules))