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
def test_savepoint_transaction(self):
    context = oslo_context.RequestContext()
    with enginefacade.writer.using(context) as session:
        session.add(self.User(name='u1'))
        session.flush()
        try:
            with enginefacade.writer.savepoint.using(context) as session:
                session.add(self.User(name='u2'))
                raise Exception('nope')
        except Exception:
            pass
        with enginefacade.writer.savepoint.using(context) as session:
            session.add(self.User(name='u3'))
        session.add(self.User(name='u4'))
    session = self.sessionmaker(autocommit=False)
    with session.begin():
        self.assertEqual([('u1',), ('u3',), ('u4',)], session.query(self.User.name).order_by(self.User.name).all())