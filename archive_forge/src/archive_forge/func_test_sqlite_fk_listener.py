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
def test_sqlite_fk_listener(self):
    engine = self._fixture(sqlite_fk=True)
    with engine.connect() as conn:
        self.assertEqual(1, conn.execute(sql.text('pragma foreign_keys')).scalars().first())
    engine = self._fixture(sqlite_fk=False)
    with engine.connect() as conn:
        self.assertEqual(0, conn.execute(sql.text('pragma foreign_keys')).scalars().first())