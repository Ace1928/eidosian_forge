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
def test_storage_quota(self):
    total = functools.reduce(lambda x, y: x + y, [f['size'] for f in self.owner1_fixtures])
    x = self.db_api.user_get_storage_usage(self.context1, self.owner_id1)
    self.assertEqual(total, x)