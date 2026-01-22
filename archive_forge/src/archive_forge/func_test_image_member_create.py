import copy
import datetime
import functools
from unittest import mock
import uuid
from oslo_db import exception as db_exception
from oslo_db.sqlalchemy import utils as sqlalchemyutils
from sqlalchemy import sql
from glance.common import exception
from glance.common import timeutils
from glance import context
from glance.db.sqlalchemy import api as db_api
from glance.db.sqlalchemy import models
from glance.tests import functional
import glance.tests.functional.db as db_tests
from glance.tests import utils as test_utils
@mock.patch.object(timeutils, 'utcnow')
def test_image_member_create(self, mock_utcnow):
    mock_utcnow.return_value = datetime.datetime.utcnow()
    memberships = self.db_api.image_member_find(self.context)
    self.assertEqual([], memberships)
    TENANT1 = str(uuid.uuid4())
    self.context.auth_token = 'user:%s:user' % TENANT1
    self.db_api.image_member_create(self.context, {'member': TENANT1, 'image_id': UUID1})
    memberships = self.db_api.image_member_find(self.context)
    self.assertEqual(1, len(memberships))
    actual = memberships[0]
    self.assertIsNotNone(actual['created_at'])
    self.assertIsNotNone(actual['updated_at'])
    actual.pop('id')
    actual.pop('created_at')
    actual.pop('updated_at')
    expected = {'member': TENANT1, 'image_id': UUID1, 'can_share': False, 'status': 'pending', 'deleted': False}
    self.assertEqual(expected, actual)