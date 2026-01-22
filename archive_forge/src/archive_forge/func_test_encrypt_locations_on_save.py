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
def test_encrypt_locations_on_save(self):
    image = self.image_factory.new_image(UUID1)
    self.image_repo.add(image)
    image.locations = self.foo_bar_location
    self.image_repo.save(image)
    db_data = self.db.image_get(self.context, UUID1)
    self.assertNotEqual(db_data['locations'], ['foo', 'bar'])
    decrypted_locations = [crypt.urlsafe_decrypt(self.crypt_key, location['url']) for location in db_data['locations']]
    self.assertEqual([location['url'] for location in self.foo_bar_location], decrypted_locations)