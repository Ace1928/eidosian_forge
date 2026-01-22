import json
import logging
import uuid
from peewee import *
from peewee import ColumnBase
from peewee import Expression
from peewee import Node
from peewee import NodeList
from peewee import __deprecated__
def contains_all(self, *items):
    return Expression(cast_jsonb(self), JSONB_CONTAINS_ALL_KEYS, Value(list(items), unpack=False))