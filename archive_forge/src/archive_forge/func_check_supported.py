from types import NoneType
from django.contrib.postgres.indexes import OpClass
from django.core.exceptions import ValidationError
from django.db import DEFAULT_DB_ALIAS, NotSupportedError
from django.db.backends.ddl_references import Expressions, Statement, Table
from django.db.models import BaseConstraint, Deferrable, F, Q
from django.db.models.expressions import Exists, ExpressionList
from django.db.models.indexes import IndexExpression
from django.db.models.lookups import PostgresOperatorLookup
from django.db.models.sql import Query
def check_supported(self, schema_editor):
    if self.include and self.index_type.lower() == 'spgist' and (not schema_editor.connection.features.supports_covering_spgist_indexes):
        raise NotSupportedError('Covering exclusion constraints using an SP-GiST index require PostgreSQL 14+.')