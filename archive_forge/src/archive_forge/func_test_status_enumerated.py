import datetime
from unittest import mock
import uuid
from oslo_config import cfg
import oslo_utils.importutils
import glance.async_
from glance.async_ import taskflow_executor
from glance.common import exception
from glance.common import timeutils
from glance import domain
import glance.tests.utils as test_utils
def test_status_enumerated(self):
    self.image_member.status = 'pending'
    self.image_member.status = 'accepted'
    self.image_member.status = 'rejected'
    self.assertRaises(ValueError, setattr, self.image_member, 'status', 'ellison')