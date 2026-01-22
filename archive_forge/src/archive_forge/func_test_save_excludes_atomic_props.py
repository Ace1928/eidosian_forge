import datetime
from unittest import mock
import uuid
from oslo_config import cfg
from oslo_db import exception as db_exc
from oslo_utils import encodeutils
from oslo_utils.fixture import uuidsentinel as uuids
from oslo_utils import timeutils
from sqlalchemy import orm as sa_orm
from glance.common import crypt
from glance.common import exception
import glance.context
import glance.db
from glance.db.sqlalchemy import api
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_save_excludes_atomic_props(self):
    fake_uuid = str(uuid.uuid4())
    image = self.image_repo.get(UUID1)
    image.extra_properties['os_glance_import_task'] = fake_uuid
    self.image_repo.save(image)
    image = self.image_repo.get(UUID1)
    self.assertNotIn('os_glance_import_task', image.extra_properties)
    self.image_repo.set_property_atomic(image, 'os_glance_import_task', fake_uuid)
    image = self.image_repo.get(UUID1)
    self.assertEqual(fake_uuid, image.extra_properties['os_glance_import_task'])
    image.extra_properties['os_glance_import_task'] = 'foo'
    self.image_repo.save(image)
    image = self.image_repo.get(UUID1)
    self.assertEqual(fake_uuid, image.extra_properties['os_glance_import_task'])
    del image.extra_properties['os_glance_import_task']
    self.image_repo.save(image)
    image = self.image_repo.get(UUID1)
    self.assertEqual(fake_uuid, image.extra_properties['os_glance_import_task'])