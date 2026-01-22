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
def clear_ordering(self, force=False, clear_default=True):
    """
        Remove any ordering settings if the current query allows it without
        side effects, set 'force' to True to clear the ordering regardless.
        If 'clear_default' is True, there will be no ordering in the resulting
        query (not even the model's default).
        """
    if not force and (self.is_sliced or self.distinct_fields or self.select_for_update):
        return
    self.order_by = ()
    self.extra_order_by = ()
    if clear_default:
        self.default_ordering = False