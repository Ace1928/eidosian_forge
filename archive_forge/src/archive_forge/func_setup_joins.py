import copy
import difflib
import functools
import sys
from collections import Counter, namedtuple
from collections.abc import Iterator, Mapping
from itertools import chain, count, product
from string import ascii_uppercase
from django.core.exceptions import FieldDoesNotExist, FieldError
from django.db import DEFAULT_DB_ALIAS, NotSupportedError, connections
from django.db.models.aggregates import Count
from django.db.models.constants import LOOKUP_SEP
from django.db.models.expressions import (
from django.db.models.fields import Field
from django.db.models.fields.related_lookups import MultiColSource
from django.db.models.lookups import Lookup
from django.db.models.query_utils import (
from django.db.models.sql.constants import INNER, LOUTER, ORDER_DIR, SINGLE
from django.db.models.sql.datastructures import BaseTable, Empty, Join, MultiJoin
from django.db.models.sql.where import AND, OR, ExtraWhere, NothingNode, WhereNode
from django.utils.functional import cached_property
from django.utils.regex_helper import _lazy_re_compile
from django.utils.tree import Node
def setup_joins(self, names, opts, alias, can_reuse=None, allow_many=True):
    """
        Compute the necessary table joins for the passage through the fields
        given in 'names'. 'opts' is the Options class for the current model
        (which gives the table we are starting from), 'alias' is the alias for
        the table to start the joining from.

        The 'can_reuse' defines the reverse foreign key joins we can reuse. It
        can be None in which case all joins are reusable or a set of aliases
        that can be reused. Note that non-reverse foreign keys are always
        reusable when using setup_joins().

        If 'allow_many' is False, then any reverse foreign key seen will
        generate a MultiJoin exception.

        Return the final field involved in the joins, the target field (used
        for any 'where' constraint), the final 'opts' value, the joins, the
        field path traveled to generate the joins, and a transform function
        that takes a field and alias and is equivalent to `field.get_col(alias)`
        in the simple case but wraps field transforms if they were included in
        names.

        The target field is the field containing the concrete value. Final
        field can be something different, for example foreign key pointing to
        that value. Final field is needed for example in some value
        conversions (convert 'obj' in fk__id=obj to pk val using the foreign
        key field for example).
        """
    joins = [alias]

    def final_transformer(field, alias):
        if not self.alias_cols:
            alias = None
        return field.get_col(alias)
    last_field_exception = None
    for pivot in range(len(names), 0, -1):
        try:
            path, final_field, targets, rest = self.names_to_path(names[:pivot], opts, allow_many, fail_on_missing=True)
        except FieldError as exc:
            if pivot == 1:
                raise
            else:
                last_field_exception = exc
        else:
            transforms = names[pivot:]
            break
    for name in transforms:

        def transform(field, alias, *, name, previous):
            try:
                wrapped = previous(field, alias)
                return self.try_transform(wrapped, name)
            except FieldError:
                if isinstance(final_field, Field) and last_field_exception:
                    raise last_field_exception
                else:
                    raise
        final_transformer = functools.partial(transform, name=name, previous=final_transformer)
        final_transformer.has_transforms = True
    for join in path:
        if join.filtered_relation:
            filtered_relation = join.filtered_relation.clone()
            table_alias = filtered_relation.alias
        else:
            filtered_relation = None
            table_alias = None
        opts = join.to_opts
        if join.direct:
            nullable = self.is_nullable(join.join_field)
        else:
            nullable = True
        connection = self.join_class(opts.db_table, alias, table_alias, INNER, join.join_field, nullable, filtered_relation=filtered_relation)
        reuse = can_reuse if join.m2m else None
        alias = self.join(connection, reuse=reuse)
        joins.append(alias)
    return JoinInfo(final_field, targets, opts, joins, path, final_transformer)