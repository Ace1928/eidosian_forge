import collections
import contextlib
import copy
import fixtures
import pickle
import sys
from unittest import mock
import warnings
from oslo_config import cfg
from oslo_context import context as oslo_context
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import MetaData
from sqlalchemy.orm import registry
from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy import String
from sqlalchemy import Table
from oslo_db import exception
from oslo_db import options
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import engines as oslo_engines
from oslo_db.sqlalchemy import orm
from oslo_db.tests import base as test_base
from oslo_db.tests.sqlalchemy import base as db_test_base
from oslo_db import warning
def test_options_not_supported(self):
    factory = enginefacade._TransactionFactory()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        factory.configure(fake1='x', connection_recycle_time=200, wrong2='y')
    self.assertEqual(1, len(w))
    self.assertTrue(issubclass(w[-1].category, warning.NotSupportedWarning))
    self.assertEqual("Configuration option(s) ['fake1', 'wrong2'] not supported", str(w[-1].message))