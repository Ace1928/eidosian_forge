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
def test_save_image_member(self):
    image_member = self.image_member_repo.get(TENANT2)
    image_member.status = 'accepted'
    self.image_member_repo.save(image_member)
    image_member_updated = self.image_member_repo.get(TENANT2)
    self.assertEqual(image_member.id, image_member_updated.id)
    self.assertEqual('accepted', image_member_updated.status)