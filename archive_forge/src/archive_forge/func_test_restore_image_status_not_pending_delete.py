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
def test_restore_image_status_not_pending_delete(self):
    image_id = uuid.uuid4()
    image = _db_fixture(image_id, name='restore_test', size=256, is_public=True, status='deleted')
    self.db.image_create(self.context, image)
    self.assertRaises(exception.Conflict, self.db.image_restore, self.context, image_id)