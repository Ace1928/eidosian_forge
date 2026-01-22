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
def conflict_statement(self, on_conflict, query):
    if not on_conflict._action:
        return
    action = on_conflict._action.lower()
    if action in ('replace', 'upsert'):
        return SQL('UPSERT')
    elif action not in ('ignore', 'nothing', 'update'):
        raise ValueError('Un-supported action for conflict resolution. CockroachDB supports REPLACE (UPSERT), IGNORE and UPDATE.')