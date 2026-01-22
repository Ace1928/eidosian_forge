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
def init_connection_state(self):
    super().init_connection_state()
    assignments = []
    if self.features.is_sql_auto_is_null_enabled:
        assignments.append('SET SQL_AUTO_IS_NULL = 0')
    if self.isolation_level:
        assignments.append('SET SESSION TRANSACTION ISOLATION LEVEL %s' % self.isolation_level.upper())
    if assignments:
        with self.cursor() as cursor:
            cursor.execute('; '.join(assignments))