from __future__ import absolute_import, print_function, division
import logging
import pytest
import petl as etl
from petl.test.helpers import ieq
@pytest.mark.skipif(bool(SKIP_PYMYSQL), reason=str(SKIP_PYMYSQL))
def test_pymysql():
    import pymysql
    connect = pymysql.connect
    dbapi_connection = connect(host=host, user=user, password=password, database=database)
    _setup_mysql(dbapi_connection)
    _test_dbo(dbapi_connection)
    _setup_mysql(dbapi_connection)
    dbapi_cursor = dbapi_connection.cursor()
    _test_dbo(dbapi_cursor)
    dbapi_cursor.close()
    _setup_mysql(dbapi_connection)
    from sqlalchemy import create_engine
    sqlalchemy_engine = create_engine('mysql+pymysql://%s:%s@%s/%s' % (user, password, host, database))
    from sqlalchemy.event import listen
    listen(sqlalchemy_engine, 'connect', _setup_sqlalchemy_quotes)
    sqlalchemy_connection = sqlalchemy_engine.connect()
    _test_dbo(sqlalchemy_connection)
    sqlalchemy_connection.close()
    _setup_mysql(dbapi_connection)
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=sqlalchemy_engine)
    sqlalchemy_session = Session()
    _test_dbo(sqlalchemy_session)
    sqlalchemy_session.close()
    _setup_mysql(dbapi_connection)
    sqlalchemy_engine2 = create_engine('mysql+pymysql://%s:%s@%s/%s' % (user, password, host, database), echo_pool='debug')
    listen(sqlalchemy_engine2, 'connect', _setup_sqlalchemy_quotes)
    _test_dbo(sqlalchemy_engine2)
    sqlalchemy_engine2.dispose()
    _test_with_schema(dbapi_connection, database)
    utf8_connection = connect(host=host, user=user, password=password, database=database, charset='utf8')
    utf8_connection.cursor().execute('SET SQL_MODE=ANSI_QUOTES')
    _test_unicode(utf8_connection)
    utf8_connection.close()