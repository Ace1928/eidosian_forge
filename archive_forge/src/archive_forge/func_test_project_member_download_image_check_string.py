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
def test_project_member_download_image_check_string(self):
    expected = "role:member and (project_id:%(project_id)s or project_id:%(member_id)s or 'community':%(visibility)s or 'public':%(visibility)s or 'shared':%(visibility)s)"
    self.assertEqual(expected, base_policy.PROJECT_MEMBER_OR_IMAGE_MEMBER_OR_COMMUNITY_OR_PUBLIC_OR_SHARED)