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
def test_policy_enforcer_does_not_raise_forbidden(self):
    self.config(enforce_scope=False, group='oslo_policy')
    self.assertTrue(self._test_enforce_scope())