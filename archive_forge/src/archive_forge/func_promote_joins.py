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
def promote_joins(self, aliases):
    """
        Promote recursively the join type of given aliases and its children to
        an outer join. If 'unconditional' is False, only promote the join if
        it is nullable or the parent join is an outer join.

        The children promotion is done to avoid join chains that contain a LOUTER
        b INNER c. So, if we have currently a INNER b INNER c and a->b is promoted,
        then we must also promote b->c automatically, or otherwise the promotion
        of a->b doesn't actually change anything in the query results.
        """
    aliases = list(aliases)
    while aliases:
        alias = aliases.pop(0)
        if self.alias_map[alias].join_type is None:
            continue
        assert self.alias_map[alias].join_type is not None
        parent_alias = self.alias_map[alias].parent_alias
        parent_louter = parent_alias and self.alias_map[parent_alias].join_type == LOUTER
        already_louter = self.alias_map[alias].join_type == LOUTER
        if (self.alias_map[alias].nullable or parent_louter) and (not already_louter):
            self.alias_map[alias] = self.alias_map[alias].promote()
            aliases.extend((join for join in self.alias_map if self.alias_map[join].parent_alias == alias and join not in aliases))