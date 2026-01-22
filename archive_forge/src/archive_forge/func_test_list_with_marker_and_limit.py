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
def test_list_with_marker_and_limit(self):
    full_tasks = self.task_repo.list()
    full_ids = [i.task_id for i in full_tasks]
    marked_tasks = self.task_repo.list(marker=full_ids[0], limit=1)
    actual_ids = [i.task_id for i in marked_tasks]
    self.assertEqual(full_ids[1:2], actual_ids)