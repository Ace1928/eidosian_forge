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
def test_new_manager_from_options(self):
    """test enginefacade's defaults given a default structure from opts"""
    factory = enginefacade._TransactionFactory()
    cfg.CONF.register_opts(options.database_opts, 'database')
    factory.configure(**dict(cfg.CONF.database.items()))
    engine_args = factory._engine_args_for_conf(None)
    self.assertEqual(None, engine_args['mysql_wsrep_sync_wait'])
    self.assertEqual(True, engine_args['sqlite_synchronous'])
    self.assertEqual('TRADITIONAL', engine_args['mysql_sql_mode'])