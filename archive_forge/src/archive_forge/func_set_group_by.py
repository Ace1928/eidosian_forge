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
def set_group_by(self, allow_aliases=True):
    """
        Expand the GROUP BY clause required by the query.

        This will usually be the set of all non-aggregate fields in the
        return data. If the database backend supports grouping by the
        primary key, and the query would be equivalent, the optimization
        will be made automatically.
        """
    if allow_aliases and self.values_select:
        group_by_annotations = {}
        values_select = {}
        for alias, expr in zip(self.values_select, self.select):
            if isinstance(expr, Col):
                values_select[alias] = expr
            else:
                group_by_annotations[alias] = expr
        self.annotations = {**group_by_annotations, **self.annotations}
        self.append_annotation_mask(group_by_annotations)
        self.select = tuple(values_select.values())
        self.values_select = tuple(values_select)
    group_by = list(self.select)
    for alias, annotation in self.annotation_select.items():
        if not (group_by_cols := annotation.get_group_by_cols()):
            continue
        if allow_aliases and (not annotation.contains_aggregate):
            group_by.append(Ref(alias, annotation))
        else:
            group_by.extend(group_by_cols)
    self.group_by = tuple(group_by)