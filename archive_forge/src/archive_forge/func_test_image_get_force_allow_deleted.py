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
def test_image_get_force_allow_deleted(self):
    self.db_api.image_destroy(self.adm_context, UUID1)
    image = self.db_api.image_get(self.context, UUID1, force_show_deleted=True)
    self.assertEqual(self.fixtures[0]['id'], image['id'])