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
def test_is_public_is_a_normal_filter_for_admin(self):
    self._setup_is_public_red_herring()
    images = self.db_api.image_get_all(self.admin_context, filters={'is_public': 'silly'})
    self.assertEqual(1, len(images))
    self.assertEqual('Red Herring', images[0]['name'])