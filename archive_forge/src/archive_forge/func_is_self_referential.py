import copy
from decimal import Decimal
from django.apps.registry import Apps
from django.db import NotSupportedError
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.backends.ddl_references import Statement
from django.db.backends.utils import strip_quotes
from django.db.models import NOT_PROVIDED, UniqueConstraint
def is_self_referential(f):
    return f.is_relation and f.remote_field.model is model