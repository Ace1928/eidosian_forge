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
def get_foreign_key_constraint_name(engine, table_name, column_name):
    """Find the name of foreign key in a table, given constrained column name.

    :param engine: a SQLAlchemy engine (or connection)

    :param table_name: name of table which contains the constraint

    :param column_name: name of column that is constrained by the foreign key.

    :return: the name of the first foreign key constraint which constrains
     the given column in the given table.

    """
    insp = inspect(engine)
    for fk in insp.get_foreign_keys(table_name):
        if column_name in fk['constrained_columns']:
            return fk['name']