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
def test_sorted_list_with_multiple_dirs(self):
    temp_id = 'd80a1a6c-bd1f-41c5-90ee-81afedb1d58d'
    image = _db_fixture(temp_id, owner=TENANT1, checksum=CHECKSUM, name='1', size=1024, is_public=True, status='active', locations=[{'url': UUID1_LOCATION, 'metadata': UUID1_LOCATION_METADATA, 'status': 'active'}])
    self.db.image_create(None, image)
    images = self.image_repo.list(sort_key=['name', 'size'], sort_dir=['asc', 'desc'])
    image_ids = [i.image_id for i in images]
    self.assertEqual([temp_id, UUID1, UUID2, UUID3], image_ids)
    images = self.image_repo.list(sort_key=['name', 'size'], sort_dir=['desc', 'asc'])
    image_ids = [i.image_id for i in images]
    self.assertEqual([UUID3, UUID2, UUID1, temp_id], image_ids)