from collections import namedtuple
import sqlparse
from MySQLdb.constants import FIELD_TYPE
from django.db.backends.base.introspection import BaseDatabaseIntrospection
from django.db.backends.base.introspection import FieldInfo as BaseFieldInfo
from django.db.backends.base.introspection import TableInfo as BaseTableInfo
from django.db.models import Index
from django.utils.datastructures import OrderedSet
def get_table_list(self, cursor):
    """Return a list of table and view names in the current database."""
    cursor.execute('\n            SELECT\n                table_name,\n                table_type,\n                table_comment\n            FROM information_schema.tables\n            WHERE table_schema = DATABASE()\n            ')
    return [TableInfo(row[0], {'BASE TABLE': 't', 'VIEW': 'v'}.get(row[1]), row[2]) for row in cursor.fetchall()]