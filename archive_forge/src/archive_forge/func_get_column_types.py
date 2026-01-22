from collections import namedtuple
from inspect import isclass
import re
import warnings
from peewee import *
from peewee import _StringField
from peewee import _query_val_transform
from peewee import CommaNodeList
from peewee import SCOPE_VALUES
from peewee import make_snake_case
from peewee import text_type
def get_column_types(self, table, schema=None):
    column_types = {}
    columns = self.database.get_columns(table)
    for column in columns:
        column_types[column.name] = self._map_col(column.data_type)
    return (column_types, {})