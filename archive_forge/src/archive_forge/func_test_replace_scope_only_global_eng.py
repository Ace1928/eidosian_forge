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
def test_replace_scope_only_global_eng(self):
    alt_connection1 = 'sqlite:///?timeout=90'
    alt_mgr1 = enginefacade.transaction_context()
    alt_mgr1.configure(connection=alt_connection1)
    alt_connection2 = 'sqlite:///?timeout=120'
    alt_mgr2 = enginefacade.transaction_context()
    alt_mgr2.configure(connection=alt_connection2)

    @enginefacade.writer
    def go1(context):
        s1 = context.session
        self.assertEqual(s1.bind.url, self.engine.url)
        with alt_mgr1.replace.using(context):
            go2(context)
        go6(context)

    @enginefacade.writer
    def go2(context):
        s2 = context.session
        self.assertEqual(str(s2.bind.url), alt_connection1)
        self.assertIs(s2.bind, alt_mgr1._factory._writer_engine)
        go3(context)

    @alt_mgr2.writer
    def go3(context):
        s3 = context.session
        self.assertEqual(str(s3.bind.url), alt_connection2)
        self.assertIs(s3.bind, alt_mgr2._factory._writer_engine)
        go4(context)

    @enginefacade.writer
    def go4(context):
        s4 = context.session
        self.assertEqual(str(s4.bind.url), alt_connection1)
        self.assertIs(s4.bind, alt_mgr1._factory._writer_engine)

    @enginefacade.writer
    def go5(context):
        s5 = context.session
        self.assertEqual(str(s5.bind.url), alt_connection1)
        self.assertIs(s5.bind, alt_mgr1._factory._writer_engine)

    @enginefacade.writer
    def go6(context):
        s6 = context.session
        self.assertEqual(str(s6.bind.url), str(self.engine.url))
        self.assertIs(s6.bind, self.engine)
    context = oslo_context.RequestContext()
    go1(context)