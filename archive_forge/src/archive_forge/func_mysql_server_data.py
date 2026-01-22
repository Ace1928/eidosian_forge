from django.core.exceptions import ImproperlyConfigured
from django.db import IntegrityError
from django.db.backends import utils as backend_utils
from django.db.backends.base.base import BaseDatabaseWrapper
from django.utils.asyncio import async_unsafe
from django.utils.functional import cached_property
from django.utils.regex_helper import _lazy_re_compile
from MySQLdb.constants import CLIENT, FIELD_TYPE
from MySQLdb.converters import conversions
from .client import DatabaseClient
from .creation import DatabaseCreation
from .features import DatabaseFeatures
from .introspection import DatabaseIntrospection
from .operations import DatabaseOperations
from .schema import DatabaseSchemaEditor
from .validation import DatabaseValidation
@cached_property
def mysql_server_data(self):
    with self.temporary_connection() as cursor:
        cursor.execute("\n                SELECT VERSION(),\n                       @@sql_mode,\n                       @@default_storage_engine,\n                       @@sql_auto_is_null,\n                       @@lower_case_table_names,\n                       CONVERT_TZ('2001-01-01 01:00:00', 'UTC', 'UTC') IS NOT NULL\n            ")
        row = cursor.fetchone()
    return {'version': row[0], 'sql_mode': row[1], 'default_storage_engine': row[2], 'sql_auto_is_null': bool(row[3]), 'lower_case_table_names': bool(row[4]), 'has_zoneinfo_database': bool(row[5])}