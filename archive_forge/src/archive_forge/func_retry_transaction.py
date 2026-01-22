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
def retry_transaction(self, max_attempts=None, system_time=None, priority=None):

    def deco(cb):

        @functools.wraps(cb)
        def new_fn():
            return run_transaction(self, cb, max_attempts, system_time, priority)
        return new_fn
    return deco