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
def test_modelbase_update(self):
    h = {'a': '1', 'b': '2'}
    self.mb.update(h)
    for key in h.keys():
        self.assertEqual(h[key], self.mb[key])