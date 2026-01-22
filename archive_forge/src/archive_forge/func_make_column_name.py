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
def make_column_name(self, column, is_foreign_key=False, snake_case=True):
    column = column.strip()
    if snake_case:
        column = make_snake_case(column)
    column = column.lower()
    if is_foreign_key:
        column = re.sub('_id$', '', column) or column
    column = re.sub('[^\\w]+', '_', column)
    if column in RESERVED_WORDS:
        column += '_'
    if len(column) and column[0].isdigit():
        column = '_' + column
    return column