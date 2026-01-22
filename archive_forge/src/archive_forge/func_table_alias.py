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
def table_alias(self, table_name, create=False, filtered_relation=None):
    """
        Return a table alias for the given table_name and whether this is a
        new alias or not.

        If 'create' is true, a new alias is always created. Otherwise, the
        most recently created alias for the table (if one exists) is reused.
        """
    alias_list = self.table_map.get(table_name)
    if not create and alias_list:
        alias = alias_list[0]
        self.alias_refcount[alias] += 1
        return (alias, False)
    if alias_list:
        alias = '%s%d' % (self.alias_prefix, len(self.alias_map) + 1)
        alias_list.append(alias)
    else:
        alias = filtered_relation.alias if filtered_relation is not None else table_name
        self.table_map[table_name] = [alias]
    self.alias_refcount[alias] = 1
    return (alias, True)