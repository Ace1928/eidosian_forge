from collections import abc
from unittest import mock
import hashlib
import os.path
import oslo_config.cfg
from oslo_policy import policy as common_policy
import glance.api.policy
from glance.common import exception
import glance.context
from glance.policies import base as base_policy
from glance.tests.unit import base
def test_context_nonadmin_policy_nonadmin(self):
    self._do_test_policy_influence_context_admin('test_admin', 'demo', False, False)