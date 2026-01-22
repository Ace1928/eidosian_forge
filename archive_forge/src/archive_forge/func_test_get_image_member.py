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
def test_get_image_member(self):
    image = self.image_repo.get(UUID1)
    image_member = self.image_member_factory.new_image_member(image, TENANT4)
    self.assertIsNone(image_member.id)
    self.image_member_repo.add(image_member)
    member = self.image_member_repo.get(image_member.member_id)
    self.assertEqual(member.id, image_member.id)
    self.assertEqual(member.image_id, image_member.image_id)
    self.assertEqual(member.member_id, image_member.member_id)
    self.assertEqual('pending', member.status)