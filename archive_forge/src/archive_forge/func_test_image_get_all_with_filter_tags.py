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
def test_image_get_all_with_filter_tags(self):
    self.db_api.image_tag_create(self.context, UUID1, 'x86')
    self.db_api.image_tag_create(self.context, UUID1, '64bit')
    self.db_api.image_tag_create(self.context, UUID2, 'power')
    self.db_api.image_tag_create(self.context, UUID2, '64bit')
    images = self.db_api.image_get_all(self.context, filters={'tags': ['64bit']})
    self.assertEqual(2, len(images))
    image_ids = [image['id'] for image in images]
    expected = [UUID1, UUID2]
    self.assertEqual(sorted(expected), sorted(image_ids))