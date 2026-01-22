import functools
import os
import fixtures
from oslo_db import options as db_options
from oslo_db.sqlalchemy import enginefacade
from keystone.common import sql
import keystone.conf
from keystone.tests import unit
def initialize_sql_session(connection_str=unit.IN_MEM_DB_CONN_STRING, enforce_sqlite_fks=True):
    db_options.set_defaults(CONF, connection=connection_str)
    facade = enginefacade.writer
    engine = facade.get_engine()
    f_key = 'ON' if enforce_sqlite_fks else 'OFF'
    if engine.name == 'sqlite':
        engine.connect().exec_driver_sql('PRAGMA foreign_keys = ' + f_key)