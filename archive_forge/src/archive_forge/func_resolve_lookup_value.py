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
def resolve_lookup_value(self, value, can_reuse, allow_joins, summarize=False):
    if hasattr(value, 'resolve_expression'):
        value = value.resolve_expression(self, reuse=can_reuse, allow_joins=allow_joins, summarize=summarize)
    elif isinstance(value, (list, tuple)):
        values = (self.resolve_lookup_value(sub_value, can_reuse, allow_joins) for sub_value in value)
        type_ = type(value)
        if hasattr(type_, '_make'):
            return type_(*values)
        return type_(values)
    return value