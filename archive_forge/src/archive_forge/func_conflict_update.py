import functools
import re
import sys
from peewee import *
from peewee import _atomic
from peewee import _manual
from peewee import ColumnMetadata  # (name, data_type, null, primary_key, table, default)
from peewee import EnclosedNodeList
from peewee import Entity
from peewee import ForeignKeyMetadata  # (column, dest_table, dest_column, table).
from peewee import IndexMetadata
from peewee import NodeList
from playhouse.pool import _PooledPostgresqlDatabase
def conflict_update(self, oc, query):
    action = oc._action.lower() if oc._action else ''
    if action in ('ignore', 'nothing'):
        parts = [SQL('ON CONFLICT')]
        if oc._conflict_target:
            parts.append(EnclosedNodeList([Entity(col) if isinstance(col, basestring) else col for col in oc._conflict_target]))
        parts.append(SQL('DO NOTHING'))
        return NodeList(parts)
    elif action in ('replace', 'upsert'):
        return
    elif oc._conflict_constraint:
        raise ValueError('CockroachDB does not support the usage of a constraint name. Use the column(s) instead.')
    return super(CockroachDatabase, self).conflict_update(oc, query)