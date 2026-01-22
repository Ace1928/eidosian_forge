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
def test_task_get_all(self):
    now = timeutils.utcnow()
    then = now + datetime.timedelta(days=365)
    image_id = str(uuid.uuid4())
    fixture1 = {'owner': self.context.owner, 'type': 'import', 'status': 'pending', 'input': '{"loc": "fake_1"}', 'result': "{'image_id': %s}" % image_id, 'message': 'blah_1', 'expires_at': then, 'created_at': now, 'updated_at': now}
    fixture2 = {'owner': self.context.owner, 'type': 'import', 'status': 'pending', 'input': '{"loc": "fake_2"}', 'result': "{'image_id': %s}" % image_id, 'message': 'blah_2', 'expires_at': then, 'created_at': now, 'updated_at': now}
    task1 = self.db_api.task_create(self.adm_context, fixture1)
    task2 = self.db_api.task_create(self.adm_context, fixture2)
    self.assertIsNotNone(task1)
    self.assertIsNotNone(task2)
    task1_id = task1['id']
    task2_id = task2['id']
    task_fixtures = {task1_id: fixture1, task2_id: fixture2}
    tasks = self.db_api.task_get_all(self.adm_context)
    self.assertEqual(2, len(tasks))
    self.assertEqual(set((tasks[0]['id'], tasks[1]['id'])), set((task1_id, task2_id)))
    for task in tasks:
        fixture = task_fixtures[task['id']]
        self.assertEqual(self.context.owner, task['owner'])
        self.assertEqual(fixture['type'], task['type'])
        self.assertEqual(fixture['status'], task['status'])
        self.assertEqual(fixture['expires_at'], task['expires_at'])
        self.assertFalse(task['deleted'])
        self.assertIsNone(task['deleted_at'])
        self.assertEqual(fixture['created_at'], task['created_at'])
        self.assertEqual(fixture['updated_at'], task['updated_at'])
        task_details_keys = ['input', 'message', 'result']
        for key in task_details_keys:
            self.assertNotIn(key, task)