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
def test_image_target_ignores_locations(self):
    image = ImageStub()
    target = glance.api.policy.ImageTarget(image)
    self.assertNotIn('locations', list(target))