from __future__ import absolute_import, print_function, division
import logging
from petl.compat import next, text_type, string_types
from petl.errors import ArgumentError
from petl.util.base import Table
from petl.io.db_utils import _is_dbapi_connection, _is_dbapi_cursor, \
from petl.io.db_create import drop_table, create_table
def todb(table, dbo, tablename, schema=None, commit=True, create=False, drop=False, constraints=True, metadata=None, dialect=None, sample=1000):
    """
    Load data into an existing database table via a DB-API 2.0
    connection or cursor. Note that the database table will be truncated,
    i.e., all existing rows will be deleted prior to inserting the new data.
    E.g.::

        >>> import petl as etl
        >>> table = [['foo', 'bar'],
        ...          ['a', 1],
        ...          ['b', 2],
        ...          ['c', 2]]
        >>> # using sqlite3
        ... import sqlite3
        >>> connection = sqlite3.connect('example.db')
        >>> # assuming table "foobar" already exists in the database
        ... etl.todb(table, connection, 'foobar')
        >>> # using psycopg2
        >>> import psycopg2
        >>> connection = psycopg2.connect('dbname=example user=postgres')
        >>> # assuming table "foobar" already exists in the database
        ... etl.todb(table, connection, 'foobar')
        >>> # using pymysql
        >>> import pymysql
        >>> connection = pymysql.connect(password='moonpie', database='thangs')
        >>> # tell MySQL to use standard quote character
        ... connection.cursor().execute('SET SQL_MODE=ANSI_QUOTES')
        >>> # load data, assuming table "foobar" already exists in the database
        ... etl.todb(table, connection, 'foobar')

    N.B., for MySQL the statement ``SET SQL_MODE=ANSI_QUOTES`` is required to
    ensure MySQL uses SQL-92 standard quote characters.

    A cursor can also be provided instead of a connection, e.g.::

        >>> import psycopg2
        >>> connection = psycopg2.connect('dbname=example user=postgres')
        >>> cursor = connection.cursor()
        >>> etl.todb(table, cursor, 'foobar')

    The parameter `dbo` may also be an SQLAlchemy engine, session or
    connection object.

    The parameter `dbo` may also be a string, in which case it is interpreted
    as the name of a file containing an :mod:`sqlite3` database.

    If ``create=True`` this function will attempt to automatically create a
    database table before loading the data. This functionality requires
    `SQLAlchemy <http://www.sqlalchemy.org/>`_ to be installed.

    **Keyword arguments:**

    table : table container
        Table data to load
    dbo : database object
        DB-API 2.0 connection, callable returning a DB-API 2.0 cursor, or
        SQLAlchemy connection, engine or session
    tablename : string
        Name of the table in the database
    schema : string
        Name of the database schema to find the table in
    commit : bool
        If True commit the changes
    create : bool
        If True attempt to create the table before loading, inferring types
        from a sample of the data (requires SQLAlchemy)
    drop : bool
        If True attempt to drop the table before recreating (only relevant if
        create=True)
    constraints : bool
        If True use length and nullable constraints (only relevant if
        create=True)
    metadata : sqlalchemy.MetaData
        Custom table metadata (only relevant if create=True)
    dialect : string
        One of {'access', 'sybase', 'sqlite', 'informix', 'firebird', 'mysql',
        'oracle', 'maxdb', 'postgresql', 'mssql'} (only relevant if
        create=True)
    sample : int
        Number of rows to sample when inferring types etc. Set to 0 to use the
        whole table (only relevant if create=True)

    .. note::

        This function is in principle compatible with any DB-API 2.0
        compliant database driver. However, at the time of writing some DB-API
        2.0 implementations, including cx_Oracle and MySQL's
        Connector/Python, are not compatible with this function, because they
        only accept a list argument to the cursor.executemany() function
        called internally by :mod:`petl`. This can be worked around by
        proxying the cursor objects, e.g.::

            >>> import cx_Oracle
            >>> connection = cx_Oracle.Connection(...)
            >>> class CursorProxy(object):
            ...     def __init__(self, cursor):
            ...         self._cursor = cursor
            ...     def executemany(self, statement, parameters, **kwargs):
            ...         # convert parameters to a list
            ...         parameters = list(parameters)
            ...         # pass through to proxied cursor
            ...         return self._cursor.executemany(statement, parameters, **kwargs)
            ...     def __getattr__(self, item):
            ...         return getattr(self._cursor, item)
            ...
            >>> def get_cursor():
            ...     return CursorProxy(connection.cursor())
            ...
            >>> import petl as etl
            >>> etl.todb(tbl, get_cursor, ...)

        Note however that this does imply loading the entire table into
        memory as a list prior to inserting into the database.

    """
    needs_closing = False
    if isinstance(dbo, string_types):
        import sqlite3
        dbo = sqlite3.connect(dbo)
        needs_closing = True
    try:
        if create:
            if drop:
                drop_table(dbo, tablename, schema=schema, commit=commit)
            create_table(table, dbo, tablename, schema=schema, commit=commit, constraints=constraints, metadata=metadata, dialect=dialect, sample=sample)
        _todb(table, dbo, tablename, schema=schema, commit=commit, truncate=True)
    finally:
        if needs_closing:
            dbo.close()