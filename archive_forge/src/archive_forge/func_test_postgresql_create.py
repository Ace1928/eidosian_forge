from __future__ import absolute_import, print_function, division
import logging
from datetime import datetime, date
import sqlite3
import pytest
from petl.io.db import fromdb, todb
from petl.io.db_create import make_sqlalchemy_column
from petl.test.helpers import ieq, eq_
from petl.util.vis import look
from petl.test.io.test_db_server import user, password, host, database
@pytest.mark.skipif(bool(SKIP_POSTGRES), reason=str(SKIP_POSTGRES))
def test_postgresql_create():
    import psycopg2
    import psycopg2.extensions
    psycopg2.extensions.register_type(psycopg2.extensions.UNICODE)
    psycopg2.extensions.register_type(psycopg2.extensions.UNICODEARRAY)
    dbapi_connection = psycopg2.connect('host=%s dbname=%s user=%s password=%s' % (host, database, user, password))
    dbapi_connection.autocommit = True
    _setup_generic(dbapi_connection)
    _test_create(dbapi_connection)
    _setup_generic(dbapi_connection)
    dbapi_cursor = dbapi_connection.cursor()
    _test_create(dbapi_cursor)
    dbapi_cursor.close()