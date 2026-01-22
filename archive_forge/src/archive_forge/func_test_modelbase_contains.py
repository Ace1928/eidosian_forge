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
def test_modelbase_contains(self):
    mb = models.ModelBase()
    h = {'a': '1', 'b': '2'}
    mb.update(h)
    for key in h.keys():
        self.assertIn(key, mb)
    self.assertNotIn('non-existent-key', mb)