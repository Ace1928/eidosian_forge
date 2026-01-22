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
def test_image_get_all_with_filter_deleted_tags(self):
    tag = self.db_api.image_tag_create(self.context, UUID1, 'AIX')
    images = self.db_api.image_get_all(self.context, filters={'tags': [tag]})
    self.assertEqual(1, len(images))
    self.db_api.image_tag_delete(self.context, UUID1, tag)
    images = self.db_api.image_get_all(self.context, filters={'tags': [tag]})
    self.assertEqual(0, len(images))