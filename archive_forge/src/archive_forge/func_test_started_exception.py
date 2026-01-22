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
def test_started_exception(self):
    facade = enginefacade.transaction_context()
    self.assertFalse(facade.is_started)
    facade.configure(connection=self.engine_uri)
    facade.writer.get_engine()
    exc = self.assertRaises(enginefacade.AlreadyStartedError, facade.configure, connection=self.engine_uri)
    self.assertEqual('this TransactionFactory is already started', exc.args[0])