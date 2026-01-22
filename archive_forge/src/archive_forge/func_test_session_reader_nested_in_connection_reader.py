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
def test_session_reader_nested_in_connection_reader(self):
    context = oslo_context.RequestContext()

    @enginefacade.reader.connection
    def go1(context):
        context.connection.execute('test1')
        go2(context)

    @enginefacade.reader
    def go2(context):
        context.session.execute('test2')
    go1(context)
    with self._assert_engines() as engines:
        with self._assert_reader_connection(engines) as connection:
            connection.execute('test1')
            with self._assert_makers(engines) as makers:
                with self._assert_reader_session(makers, connection) as session:
                    session.execute('test2')