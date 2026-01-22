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
def test_logs_real_mode(self):
    log = self.useFixture(fixtures.FakeLogger(level=logging.DEBUG))
    engine = self._fixture(sql_mode='TRADITIONAL')
    with engine.connect() as conn:
        actual_mode = conn.execute(sql.text("SHOW VARIABLES LIKE 'sql_mode'")).fetchone()[1]
    self.assertIn('MySQL server mode set to %s' % actual_mode, log.output)