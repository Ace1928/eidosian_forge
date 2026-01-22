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
def test_new_image_with_extra_properties_and_tags(self):
    extra_properties = {'foo': 'bar'}
    tags = ['one', 'two']
    image = self.image_factory.new_image(image_id=UUID1, name='image-1', extra_properties=extra_properties, tags=tags)
    self.assertEqual(UUID1, image.image_id, UUID1)
    self.assertIsNotNone(image.created_at)
    self.assertEqual(image.created_at, image.updated_at)
    self.assertEqual('queued', image.status)
    self.assertEqual('shared', image.visibility)
    self.assertIsNone(image.owner)
    self.assertEqual('image-1', image.name)
    self.assertIsNone(image.size)
    self.assertEqual(0, image.min_disk)
    self.assertEqual(0, image.min_ram)
    self.assertFalse(image.protected)
    self.assertIsNone(image.disk_format)
    self.assertIsNone(image.container_format)
    self.assertEqual({'foo': 'bar'}, image.extra_properties)
    self.assertEqual(set(['one', 'two']), image.tags)