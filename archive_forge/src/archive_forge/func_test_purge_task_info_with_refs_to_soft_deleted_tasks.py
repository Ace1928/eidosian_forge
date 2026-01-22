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
def test_purge_task_info_with_refs_to_soft_deleted_tasks(self):
    session = db_api.get_session()
    engine = db_api.get_engine()
    tasks = self.db_api.task_get_all(self.adm_context)
    self.assertEqual(3, len(tasks))
    task_info = sqlalchemyutils.get_table(engine, 'task_info')
    with session.begin():
        task_info_rows = session.query(task_info).count()
    self.assertEqual(3, task_info_rows)
    self.db_api.purge_deleted_rows(self.context, 1, 5)
    tasks = self.db_api.task_get_all(self.adm_context)
    self.assertEqual(2, len(tasks))
    with session.begin():
        task_info_rows = session.query(task_info).count()
    self.assertEqual(2, task_info_rows)