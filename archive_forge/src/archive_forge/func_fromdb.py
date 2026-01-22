from __future__ import absolute_import, print_function, division
import logging
from petl.compat import next, text_type, string_types
from petl.errors import ArgumentError
from petl.util.base import Table
from petl.io.db_utils import _is_dbapi_connection, _is_dbapi_cursor, \
from petl.io.db_create import drop_table, create_table
def fromdb(dbo, query, *args, **kwargs):
    """Provides access to data from any DB-API 2.0 connection via a given query.
    E.g., using :mod:`sqlite3`::

        >>> import petl as etl
        >>> import sqlite3
        >>> connection = sqlite3.connect('example.db')
        >>> table = etl.fromdb(connection, 'SELECT * FROM example')

    E.g., using :mod:`psycopg2` (assuming you've installed it first)::

        >>> import petl as etl
        >>> import psycopg2
        >>> connection = psycopg2.connect('dbname=example user=postgres')
        >>> table = etl.fromdb(connection, 'SELECT * FROM example')

    E.g., using :mod:`pymysql` (assuming you've installed it first)::

        >>> import petl as etl
        >>> import pymysql
        >>> connection = pymysql.connect(password='moonpie', database='thangs')
        >>> table = etl.fromdb(connection, 'SELECT * FROM example')

    The `dbo` argument may also be a function that creates a cursor. N.B., each
    call to the function should return a new cursor. E.g.::

        >>> import petl as etl
        >>> import psycopg2
        >>> connection = psycopg2.connect('dbname=example user=postgres')
        >>> mkcursor = lambda: connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
        >>> table = etl.fromdb(mkcursor, 'SELECT * FROM example')

    The parameter `dbo` may also be an SQLAlchemy engine, session or
    connection object.

    The parameter `dbo` may also be a string, in which case it is interpreted as
    the name of a file containing an :mod:`sqlite3` database.

    Note that the default behaviour of most database servers and clients is for
    the entire result set for each query to be sent from the server to the
    client. If your query returns a large result set this can result in
    significant memory usage at the client. Some databases support server-side
    cursors which provide a means for client libraries to fetch result sets
    incrementally, reducing memory usage at the client.

    To use a server-side cursor with a PostgreSQL database, e.g.::

        >>> import petl as etl
        >>> import psycopg2
        >>> connection = psycopg2.connect('dbname=example user=postgres')
        >>> table = etl.fromdb(lambda: connection.cursor(name='arbitrary'),
        ...                    'SELECT * FROM example')

    For more information on server-side cursors see the following links:

        * http://initd.org/psycopg/docs/usage.html#server-side-cursors
        * http://mysql-python.sourceforge.net/MySQLdb.html#using-and-extending

    """
    if isinstance(dbo, string_types):
        import sqlite3
        dbo = sqlite3.connect(dbo)
    return DbView(dbo, query, *args, **kwargs)