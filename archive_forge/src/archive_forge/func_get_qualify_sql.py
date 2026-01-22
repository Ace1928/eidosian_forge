import collections
import json
import re
from functools import partial
from itertools import chain
from django.core.exceptions import EmptyResultSet, FieldError, FullResultSet
from django.db import DatabaseError, NotSupportedError
from django.db.models.constants import LOOKUP_SEP
from django.db.models.expressions import F, OrderBy, RawSQL, Ref, Value
from django.db.models.functions import Cast, Random
from django.db.models.lookups import Lookup
from django.db.models.query_utils import select_related_descend
from django.db.models.sql.constants import (
from django.db.models.sql.query import Query, get_order_dir
from django.db.models.sql.where import AND
from django.db.transaction import TransactionManagementError
from django.utils.functional import cached_property
from django.utils.hashable import make_hashable
from django.utils.regex_helper import _lazy_re_compile
def get_qualify_sql(self):
    where_parts = []
    if self.where:
        where_parts.append(self.where)
    if self.having:
        where_parts.append(self.having)
    inner_query = self.query.clone()
    inner_query.subquery = True
    inner_query.where = inner_query.where.__class__(where_parts)
    select = {expr: alias for expr, _, alias in self.get_select(with_col_aliases=True)[0]}
    select_aliases = set(select.values())
    qual_aliases = set()
    replacements = {}

    def collect_replacements(expressions):
        while expressions:
            expr = expressions.pop()
            if expr in replacements:
                continue
            elif (select_alias := select.get(expr)):
                replacements[expr] = select_alias
            elif isinstance(expr, Lookup):
                expressions.extend(expr.get_source_expressions())
            elif isinstance(expr, Ref):
                if expr.refs not in select_aliases:
                    expressions.extend(expr.get_source_expressions())
            else:
                num_qual_alias = len(qual_aliases)
                select_alias = f'qual{num_qual_alias}'
                qual_aliases.add(select_alias)
                inner_query.add_annotation(expr, select_alias)
                replacements[expr] = select_alias
    collect_replacements(list(self.qualify.leaves()))
    self.qualify = self.qualify.replace_expressions({expr: Ref(alias, expr) for expr, alias in replacements.items()})
    order_by = []
    for order_by_expr, *_ in self.get_order_by():
        collect_replacements(order_by_expr.get_source_expressions())
        order_by.append(order_by_expr.replace_expressions({expr: Ref(alias, expr) for expr, alias in replacements.items()}))
    inner_query_compiler = inner_query.get_compiler(self.using, connection=self.connection, elide_empty=self.elide_empty)
    inner_sql, inner_params = inner_query_compiler.as_sql(with_limits=False, with_col_aliases=True)
    qualify_sql, qualify_params = self.compile(self.qualify)
    result = ['SELECT * FROM (', inner_sql, ')', self.connection.ops.quote_name('qualify'), 'WHERE', qualify_sql]
    if qual_aliases:
        cols = [self.connection.ops.quote_name(alias) for alias in select.values()]
        result = ['SELECT', ', '.join(cols), 'FROM (', *result, ')', self.connection.ops.quote_name('qualify_mask')]
    params = list(inner_params) + qualify_params
    if order_by:
        ordering_sqls = []
        for ordering in order_by:
            ordering_sql, ordering_params = self.compile(ordering)
            ordering_sqls.append(ordering_sql)
            params.extend(ordering_params)
        result.extend(['ORDER BY', ', '.join(ordering_sqls)])
    return (result, params)