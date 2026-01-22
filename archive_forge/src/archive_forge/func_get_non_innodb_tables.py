import collections
from collections import abc
import itertools
import logging
import re
from oslo_utils import timeutils
import sqlalchemy
from sqlalchemy import Boolean
from sqlalchemy.engine import Connectable
from sqlalchemy.engine import url as sa_url
from sqlalchemy import exc
from sqlalchemy import func
from sqlalchemy import Index
from sqlalchemy import inspect
from sqlalchemy import Integer
from sqlalchemy import MetaData
from sqlalchemy.sql.expression import cast
from sqlalchemy.sql.expression import literal_column
from sqlalchemy.sql import text
from sqlalchemy import Table
from oslo_db._i18n import _
from oslo_db import exception
from oslo_db.sqlalchemy import models
def get_non_innodb_tables(connectable, skip_tables=('migrate_version', 'alembic_version')):
    """Get a list of tables which don't use InnoDB storage engine.

    :param connectable: a SQLAlchemy Engine or a Connection instance
    :param skip_tables: a list of tables which might have a different
                        storage engine
    """
    query_str = "\n        SELECT table_name\n        FROM information_schema.tables\n        WHERE table_schema = :database AND\n              engine != 'InnoDB'\n    "
    params = {}
    if skip_tables:
        params = dict((('skip_%s' % i, table_name) for i, table_name in enumerate(skip_tables)))
        placeholders = ', '.join((':' + p for p in params))
        query_str += ' AND table_name NOT IN (%s)' % placeholders
    params['database'] = connectable.engine.url.database
    query = text(query_str)
    with connectable.connect() as conn, conn.begin():
        noninnodb = conn.execute(query, params)
    return [i[0] for i in noninnodb]