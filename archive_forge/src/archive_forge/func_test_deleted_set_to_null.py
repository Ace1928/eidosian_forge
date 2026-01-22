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
def test_deleted_set_to_null(self):
    m = SoftDeletedModel(id=123456, smth='test')
    self.session.add(m)
    self.session.commit()
    m.deleted = None
    self.session.commit()
    self.assertIsNone(m.deleted)