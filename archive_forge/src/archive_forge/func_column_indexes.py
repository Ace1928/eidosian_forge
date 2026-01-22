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
def column_indexes(self, table):
    accum = {}
    for index in self.indexes[table]:
        if len(index.columns) == 1:
            accum[index.columns[0]] = index.unique
    return accum