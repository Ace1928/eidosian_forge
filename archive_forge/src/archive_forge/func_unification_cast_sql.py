import json
from functools import lru_cache, partial
from django.conf import settings
from django.db.backends.base.operations import BaseDatabaseOperations
from django.db.backends.postgresql.psycopg_any import (
from django.db.backends.utils import split_tzname_delta
from django.db.models.constants import OnConflict
from django.db.models.functions import Cast
from django.utils.regex_helper import _lazy_re_compile
def unification_cast_sql(self, output_field):
    internal_type = output_field.get_internal_type()
    if internal_type in ('GenericIPAddressField', 'IPAddressField', 'TimeField', 'UUIDField'):
        return 'CAST(%%s AS %s)' % output_field.db_type(self.connection).split('(')[0]
    return '%s'