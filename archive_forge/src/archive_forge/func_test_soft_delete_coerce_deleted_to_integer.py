from collections import abc
import datetime
from unittest import mock
from sqlalchemy import Column
from sqlalchemy import Integer, String
from sqlalchemy import event
from sqlalchemy.orm import declarative_base
from oslo_db.sqlalchemy import models
from oslo_db.tests import base as test_base
from oslo_db.tests.sqlalchemy import base as db_test_base
def test_soft_delete_coerce_deleted_to_integer(self):

    def listener(conn, cur, stmt, params, context, executemany):
        if 'insert' in stmt.lower():
            self.assertNotIn('False', str(params))
    event.listen(self.engine, 'before_cursor_execute', listener)
    self.addCleanup(event.remove, self.engine, 'before_cursor_execute', listener)
    m = SoftDeletedModel(id=1, smth='test', deleted=False)
    self.session.add(m)
    self.session.commit()