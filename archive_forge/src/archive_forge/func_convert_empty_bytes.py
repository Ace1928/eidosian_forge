import datetime
import uuid
from functools import lru_cache
from django.conf import settings
from django.db import DatabaseError, NotSupportedError
from django.db.backends.base.operations import BaseDatabaseOperations
from django.db.backends.utils import split_tzname_delta, strip_quotes, truncate_name
from django.db.models import AutoField, Exists, ExpressionWrapper, Lookup
from django.db.models.expressions import RawSQL
from django.db.models.sql.where import WhereNode
from django.utils import timezone
from django.utils.encoding import force_bytes, force_str
from django.utils.functional import cached_property
from django.utils.regex_helper import _lazy_re_compile
from .base import Database
from .utils import BulkInsertMapper, InsertVar, Oracle_datetime
@staticmethod
def convert_empty_bytes(value, expression, connection):
    return b'' if value is None else value