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
def test_admin_finding_owner1s_image_members(self):
    """Admin should see all memberships of owner1's image """
    expected = [self.tenant1, self.tenant2]
    image_id = self.image_ids[self.owner1, 'shared-with-both']
    self._check_by_image(self.admin_ctx, image_id, expected)