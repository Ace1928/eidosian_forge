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
def test_sqlite_memory_pool_args(self):
    for _url in ('sqlite://', 'sqlite:///:memory:'):
        engines._init_connection_args(utils.make_url(_url), self.args, {'max_pool_size': 10, 'max_overflow': 10})
        self.assertNotIn('pool_size', self.args)
        self.assertNotIn('max_overflow', self.args)
        self.assertEqual(False, self.args['connect_args']['check_same_thread'])
        self.assertIn('poolclass', self.args)