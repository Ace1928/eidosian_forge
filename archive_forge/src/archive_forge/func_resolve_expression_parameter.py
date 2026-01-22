import json
import warnings
from django import forms
from django.core import checks, exceptions
from django.db import NotSupportedError, connections, router
from django.db.models import expressions, lookups
from django.db.models.constants import LOOKUP_SEP
from django.db.models.fields import TextField
from django.db.models.lookups import (
from django.utils.deprecation import RemovedInDjango51Warning
from django.utils.translation import gettext_lazy as _
from . import Field
from .mixins import CheckFieldDefaultMixin
def resolve_expression_parameter(self, compiler, connection, sql, param):
    sql, params = super().resolve_expression_parameter(compiler, connection, sql, param)
    if not hasattr(param, 'as_sql') and (not connection.features.has_native_json_field):
        if connection.vendor == 'oracle':
            value = json.loads(param)
            sql = "%s(JSON_OBJECT('value' VALUE %%s FORMAT JSON), '$.value')"
            if isinstance(value, (list, dict)):
                sql %= 'JSON_QUERY'
            else:
                sql %= 'JSON_VALUE'
        elif connection.vendor == 'mysql' or (connection.vendor == 'sqlite' and params[0] not in connection.ops.jsonfield_datatype_values):
            sql = "JSON_EXTRACT(%s, '$')"
    if connection.vendor == 'mysql' and connection.mysql_is_mariadb:
        sql = 'JSON_UNQUOTE(%s)' % sql
    return (sql, params)