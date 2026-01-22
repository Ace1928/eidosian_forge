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
def test_decrypt_locations_on_list(self):
    url_loc = ['ping', 'pong']
    orig_locations = [{'url': location, 'metadata': {}, 'status': 'active'} for location in url_loc]
    encrypted_locs = [crypt.urlsafe_encrypt(self.crypt_key, location) for location in url_loc]
    encrypted_locations = [{'url': location, 'metadata': {}, 'status': 'active'} for location in encrypted_locs]
    self.assertNotEqual(encrypted_locations, orig_locations)
    db_data = _db_fixture(UUID1, owner=TENANT1, locations=encrypted_locations)
    self.db.image_create(None, db_data)
    image = self.image_repo.list()[0]
    self.assertIn('id', image.locations[0])
    self.assertIn('id', image.locations[1])
    image.locations[0].pop('id')
    image.locations[1].pop('id')
    self.assertEqual(orig_locations, image.locations)