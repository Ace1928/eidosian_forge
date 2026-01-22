import uuid
from django.conf import settings
from django.db.backends.base.operations import BaseDatabaseOperations
from django.db.backends.utils import split_tzname_delta
from django.db.models import Exists, ExpressionWrapper, Lookup
from django.db.models.constants import OnConflict
from django.utils import timezone
from django.utils.encoding import force_str
from django.utils.regex_helper import _lazy_re_compile
def time_trunc_sql(self, lookup_type, sql, params, tzname=None):
    sql, params = self._convert_sql_to_tz(sql, params, tzname)
    fields = {'hour': '%H:00:00', 'minute': '%H:%i:00', 'second': '%H:%i:%s'}
    if lookup_type in fields:
        format_str = fields[lookup_type]
        return (f'CAST(DATE_FORMAT({sql}, %s) AS TIME)', (*params, format_str))
    else:
        return (f'TIME({sql})', params)