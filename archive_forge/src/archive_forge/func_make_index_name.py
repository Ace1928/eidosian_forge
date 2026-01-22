from collections import namedtuple
import functools
import hashlib
import re
from peewee import *
from peewee import CommaNodeList
from peewee import EnclosedNodeList
from peewee import Entity
from peewee import Expression
from peewee import Node
from peewee import NodeList
from peewee import OP
from peewee import callable_
from peewee import sort_models
from peewee import sqlite3
from peewee import _truncate_constraint_name
def make_index_name(table_name, columns):
    index_name = '_'.join((table_name,) + tuple(columns))
    if len(index_name) > 64:
        index_hash = hashlib.md5(index_name.encode('utf-8')).hexdigest()
        index_name = '%s_%s' % (index_name[:56], index_hash[:7])
    return index_name