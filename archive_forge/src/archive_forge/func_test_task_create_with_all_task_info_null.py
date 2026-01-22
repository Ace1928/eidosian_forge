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
def test_task_create_with_all_task_info_null(self):
    task_id = str(uuid.uuid4())
    self.context.project_id = str(uuid.uuid4())
    values = {'id': task_id, 'owner': self.context.owner, 'type': 'export', 'status': 'pending', 'input': None, 'result': None, 'message': None}
    task_values = build_task_fixture(**values)
    task = self.db_api.task_create(self.adm_context, task_values)
    self.assertIsNotNone(task)
    self.assertEqual(task_id, task['id'])
    self.assertEqual(self.context.owner, task['owner'])
    self.assertEqual('export', task['type'])
    self.assertEqual('pending', task['status'])
    self.assertIsNone(task['input'])
    self.assertIsNone(task['result'])
    self.assertIsNone(task['message'])