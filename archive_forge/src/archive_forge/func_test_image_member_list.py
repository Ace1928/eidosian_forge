import datetime
from unittest import mock
import glance_store
from oslo_config import cfg
import oslo_messaging
import webob
import glance.async_
from glance.common import exception
from glance.common import timeutils
import glance.context
from glance import notifier
import glance.tests.unit.utils as unit_test_utils
from glance.tests import utils
def test_image_member_list(self):
    image_members = self.image_member_repo_proxy.list()
    self.assertIsInstance(image_members[0], glance.notifier.ImageMemberProxy)
    self.assertEqual('image_members_from_list', image_members[0].repo)