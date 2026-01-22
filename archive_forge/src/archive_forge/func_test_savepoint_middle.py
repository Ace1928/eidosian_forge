import logging
import os
from unittest import mock
import fixtures
from oslo_config import cfg
import sqlalchemy
from sqlalchemy.engine import base as base_engine
from sqlalchemy import exc
from sqlalchemy.pool import NullPool
from sqlalchemy import sql
from sqlalchemy import Column, MetaData, Table
from sqlalchemy import Integer, String
from sqlalchemy.orm import declarative_base
from oslo_db import exception
from oslo_db import options as db_options
from oslo_db.sqlalchemy import compat
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import engines
from oslo_db.sqlalchemy import models
from oslo_db.sqlalchemy import session
from oslo_db.sqlalchemy import utils
from oslo_db.tests import base as test_base
from oslo_db.tests.sqlalchemy import base as db_test_base
def test_savepoint_middle(self):
    with self.engine.begin() as conn:
        conn.execute(self.test_table.insert(), {'data': 'data 1'})
        savepoint = conn.begin_nested()
        conn.execute(self.test_table.insert(), {'data': 'data 2'})
        savepoint.rollback()
        conn.execute(self.test_table.insert(), {'data': 'data 3'})
        self.assertEqual([(1, 'data 1'), (2, 'data 3')], conn.execute(self.test_table.select().order_by(self.test_table.c.id)).fetchall())