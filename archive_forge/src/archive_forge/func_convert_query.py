import datetime
import decimal
import warnings
from collections.abc import Mapping
from itertools import chain, tee
from sqlite3 import dbapi2 as Database
from django.core.exceptions import ImproperlyConfigured
from django.db import IntegrityError
from django.db.backends.base.base import BaseDatabaseWrapper
from django.utils.asyncio import async_unsafe
from django.utils.dateparse import parse_date, parse_datetime, parse_time
from django.utils.regex_helper import _lazy_re_compile
from ._functions import register as register_functions
from .client import DatabaseClient
from .creation import DatabaseCreation
from .features import DatabaseFeatures
from .introspection import DatabaseIntrospection
from .operations import DatabaseOperations
from .schema import DatabaseSchemaEditor
def convert_query(self, query, *, param_names=None):
    if param_names is None:
        return FORMAT_QMARK_REGEX.sub('?', query).replace('%%', '%')
    else:
        return query % {name: f':{name}' for name in param_names}