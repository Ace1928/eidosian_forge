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
def test_modelbase_has_dict_methods(self):
    dict_methods = ('__getitem__', '__setitem__', '__contains__', 'get', 'update', 'save', 'items', 'iteritems', 'keys')
    for method in dict_methods:
        self.assertTrue(hasattr(models.ModelBase, method), 'Method %s() is not found' % method)