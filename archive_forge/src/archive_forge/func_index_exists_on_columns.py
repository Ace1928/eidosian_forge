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
def index_exists_on_columns(engine, table_name, columns):
    """Check if an index on given columns exists.

    :param engine: sqlalchemy engine
    :param table_name: name of the table
    :param columns: a list type of columns that will be checked
    """
    if not isinstance(columns, list):
        columns = list(columns)
    for index in get_indexes(engine, table_name):
        if index['column_names'] == columns:
            return True
    return False