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
def constraint_sql(self, model, schema_editor):
    query = Query(model, alias_cols=False)
    compiler = query.get_compiler(connection=schema_editor.connection)
    expressions = self._get_expressions(schema_editor, query)
    table = model._meta.db_table
    condition = self._get_condition_sql(compiler, schema_editor, query)
    include = [model._meta.get_field(field_name).column for field_name in self.include]
    return Statement(self.template, table=Table(table, schema_editor.quote_name), name=schema_editor.quote_name(self.name), index_type=self.index_type, expressions=Expressions(table, expressions, compiler, schema_editor.quote_value), where=' WHERE (%s)' % condition if condition else '', include=schema_editor._index_include_sql(model, include), deferrable=schema_editor._deferrable_constraint_sql(self.deferrable))