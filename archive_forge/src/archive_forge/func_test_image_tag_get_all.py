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
def test_image_tag_get_all(self):
    self.db_api.image_tag_create(self.context, UUID1, 'snap')
    self.db_api.image_tag_create(self.context, UUID1, 'snarf')
    self.db_api.image_tag_create(self.context, UUID2, 'snarf')
    tags = self.db_api.image_tag_get_all(self.context, UUID1)
    expected = ['snap', 'snarf']
    self.assertEqual(expected, tags)
    tags = self.db_api.image_tag_get_all(self.context, UUID2)
    expected = ['snarf']
    self.assertEqual(expected, tags)