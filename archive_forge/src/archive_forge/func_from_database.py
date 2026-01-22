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
@classmethod
def from_database(cls, database):
    if CockroachDatabase and isinstance(database, CockroachDatabase):
        return CockroachDBMigrator(database)
    elif isinstance(database, PostgresqlDatabase):
        return PostgresqlMigrator(database)
    elif isinstance(database, MySQLDatabase):
        return MySQLMigrator(database)
    elif isinstance(database, SqliteDatabase):
        return SqliteMigrator(database)
    raise ValueError('Unsupported database: %s' % database)