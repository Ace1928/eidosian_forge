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
def test_new_image_for_is_public(self):
    extra_prop = {'is_public': True}
    new_image = self.image_factory.new_image(image_id=UUID1, extra_properties=extra_prop)
    self.assertEqual(True, new_image.extra_properties['is_public'])