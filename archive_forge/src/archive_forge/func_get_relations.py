from collections import namedtuple
import sqlparse
from MySQLdb.constants import FIELD_TYPE
from django.db.backends.base.introspection import BaseDatabaseIntrospection
from django.db.backends.base.introspection import FieldInfo as BaseFieldInfo
from django.db.backends.base.introspection import TableInfo as BaseTableInfo
from django.db.models import Index
from django.utils.datastructures import OrderedSet
def get_relations(self, cursor, table_name):
    """
        Return a dictionary of {field_name: (field_name_other_table, other_table)}
        representing all foreign keys in the given table.
        """
    cursor.execute('\n            SELECT column_name, referenced_column_name, referenced_table_name\n            FROM information_schema.key_column_usage\n            WHERE table_name = %s\n                AND table_schema = DATABASE()\n                AND referenced_table_schema = DATABASE()\n                AND referenced_table_name IS NOT NULL\n                AND referenced_column_name IS NOT NULL\n            ', [table_name])
    return {field_name: (other_field, other_table) for field_name, other_field, other_table in cursor.fetchall()}