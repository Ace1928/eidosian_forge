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
def test_new_manager_from_config(self):
    normal_mgr = enginefacade.transaction_context()
    normal_mgr.configure(connection='sqlite://', sqlite_fk=True, mysql_sql_mode='FOOBAR', max_overflow=38)
    normal_mgr._factory._start()
    copied_mgr = normal_mgr.make_new_manager()
    self.assertTrue(normal_mgr._factory._started)
    self.assertIsNotNone(normal_mgr._factory._writer_engine)
    self.assertIsNot(copied_mgr._factory, normal_mgr._factory)
    self.assertFalse(copied_mgr._factory._started)
    copied_mgr._factory._start()
    self.assertIsNot(normal_mgr._factory._writer_engine, copied_mgr._factory._writer_engine)
    engine_args = copied_mgr._factory._engine_args_for_conf(None)
    self.assertTrue(engine_args['sqlite_fk'])
    self.assertEqual('FOOBAR', engine_args['mysql_sql_mode'])
    self.assertEqual(38, engine_args['max_overflow'])
    self.assertNotIn('mysql_wsrep_sync_wait', engine_args)