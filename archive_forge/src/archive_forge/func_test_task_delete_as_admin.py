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
def test_task_delete_as_admin(self):
    task_values = build_task_fixture(owner=self.context.owner)
    task = self.db_api.task_create(self.adm_context, task_values)
    self.assertIsNotNone(task)
    self.assertFalse(task['deleted'])
    self.assertIsNone(task['deleted_at'])
    task_id = task['id']
    self.db_api.task_delete(self.adm_context, task_id)
    del_task = self.db_api.task_get(self.adm_context, task_id, force_show_deleted=True)
    self.assertIsNotNone(del_task)
    self.assertEqual(task_id, del_task['id'])
    self.assertTrue(del_task['deleted'])
    self.assertIsNotNone(del_task['deleted_at'])