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
def test_modelbase_iter(self):
    expected = {'id': None, 'smth': None, 'name': 'NAME'}
    i = iter(self.ekm)
    found_items = 0
    while True:
        r = next(i, None)
        if r is None:
            break
        self.assertEqual(expected[r[0]], r[1])
        found_items += 1
    self.assertEqual(len(expected), found_items)