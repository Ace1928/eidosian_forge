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
def change_index_columns(engine, table_name, index_name, new_columns):
    """Change set of columns that are indexed by given index.

    :param engine:      sqlalchemy engine
    :param table_name:  name of the table
    :param index_name:  name of the index
    :param new_columns: tuple with names of columns that will be indexed
    """
    drop_index(engine, table_name, index_name)
    add_index(engine, table_name, index_name, new_columns)