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
def test_image_member_delete(self):
    TENANT1 = str(uuid.uuid4())
    self.context.auth_token = 'user:%s:user' % TENANT1
    fixture = {'member': TENANT1, 'image_id': UUID1, 'can_share': True}
    member = self.db_api.image_member_create(self.context, fixture)
    self.assertEqual(1, len(self.db_api.image_member_find(self.context)))
    member = self.db_api.image_member_delete(self.context, member['id'])
    self.assertEqual(0, len(self.db_api.image_member_find(self.context)))