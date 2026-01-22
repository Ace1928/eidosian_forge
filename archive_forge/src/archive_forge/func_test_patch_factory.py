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
def test_patch_factory(self):
    normal_mgr = enginefacade.transaction_context()
    normal_mgr.configure(connection='sqlite:///foo.db')
    alt_mgr = enginefacade.transaction_context()
    alt_mgr.configure(connection='sqlite:///bar.db')

    @normal_mgr.writer
    def go1(context):
        s1 = context.session
        self.assertEqual(s1.bind.url, 'sqlite:///foo.db')
        self.assertIs(s1.bind, normal_mgr._factory._writer_engine)

    @normal_mgr.writer
    def go2(context):
        s1 = context.session
        self.assertEqual(s1.bind.url, 'sqlite:///bar.db')
        self.assertIs(normal_mgr._factory._writer_engine, alt_mgr._factory._writer_engine)

    def create_engine(sql_connection, **kw):
        return mock.Mock(url=sql_connection)
    with mock.patch('oslo_db.sqlalchemy.engines.create_engine', create_engine):
        context = oslo_context.RequestContext()
        go1(context)
        reset = normal_mgr.patch_factory(alt_mgr._factory)
        go2(context)
        reset()
        go1(context)