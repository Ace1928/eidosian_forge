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
def test_sorted_list(self):
    tasks = self.task_repo.list(sort_key='status', sort_dir='desc')
    task_ids = [i.task_id for i in tasks]
    self.assertEqual([UUID2, UUID1, UUID3], task_ids)