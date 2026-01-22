import datetime
import logging
from petl.compat import long, text_type
from petl.errors import ArgumentError
from petl.util.materialise import columns
from petl.transform.basics import head
from petl.io.db_utils import _is_dbapi_connection, _is_dbapi_cursor, \
def make_create_table_statement(table, tablename, schema=None, constraints=True, metadata=None, dialect=None):
    """
    Generate a CREATE TABLE statement based on data in `table`.

    Keyword arguments:

    table : table container
        Table data to use to infer types etc.
    tablename : text
        Name of the table
    schema : text
        Name of the database schema to create the table in
    constraints : bool
        If True use length and nullable constraints
    metadata : sqlalchemy.MetaData
        Custom table metadata
    dialect : text
        One of {'access', 'sybase', 'sqlite', 'informix', 'firebird', 'mysql',
        'oracle', 'maxdb', 'postgresql', 'mssql'}

    """
    import sqlalchemy
    sql_table = make_sqlalchemy_table(table, tablename, schema=schema, constraints=constraints, metadata=metadata)
    if dialect:
        module = __import__('sqlalchemy.dialects.%s' % DIALECTS[dialect], fromlist=['dialect'])
        sql_dialect = module.dialect()
    else:
        sql_dialect = None
    return text_type(sqlalchemy.schema.CreateTable(sql_table).compile(dialect=sql_dialect)).strip()