import csv
import datetime
from decimal import Decimal
import json
import operator
import sys
import uuid
from peewee import *
from playhouse.db_url import connect
from playhouse.migrate import migrate
from playhouse.migrate import SchemaMigrator
from playhouse.reflection import Introspector
def get_table_dependencies(self, table):
    stack = [table]
    accum = []
    seen = set()
    while stack:
        table = stack.pop()
        for fk_meta in self._database.get_foreign_keys(table):
            dest = fk_meta.dest_table
            if dest not in seen:
                stack.append(dest)
                accum.append(dest)
    return accum