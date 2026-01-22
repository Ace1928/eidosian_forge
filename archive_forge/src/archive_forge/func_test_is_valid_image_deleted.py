import os
from unittest import mock
import glance_store
from oslo_config import cfg
from oslo_utils.fixture import uuidsentinel as uuids
from glance.common import exception
from glance import context
from glance import housekeeping
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_is_valid_image_deleted(self):
    image = self.db.image_create(self.context, {'status': 'queued'})
    self.db.image_destroy(self.context, image['id'])
    self.assertFalse(self.cleaner.is_valid_image(image['id']))