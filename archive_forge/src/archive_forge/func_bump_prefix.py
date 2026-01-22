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
def bump_prefix(self, other_query, exclude=None):
    """
        Change the alias prefix to the next letter in the alphabet in a way
        that the other query's aliases and this query's aliases will not
        conflict. Even tables that previously had no alias will get an alias
        after this call. To prevent changing aliases use the exclude parameter.
        """

    def prefix_gen():
        """
            Generate a sequence of characters in alphabetical order:
                -> 'A', 'B', 'C', ...

            When the alphabet is finished, the sequence will continue with the
            Cartesian product:
                -> 'AA', 'AB', 'AC', ...
            """
        alphabet = ascii_uppercase
        prefix = chr(ord(self.alias_prefix) + 1)
        yield prefix
        for n in count(1):
            seq = alphabet[alphabet.index(prefix):] if prefix else alphabet
            for s in product(seq, repeat=n):
                yield ''.join(s)
            prefix = None
    if self.alias_prefix != other_query.alias_prefix:
        return
    local_recursion_limit = sys.getrecursionlimit() // 16
    for pos, prefix in enumerate(prefix_gen()):
        if prefix not in self.subq_aliases:
            self.alias_prefix = prefix
            break
        if pos > local_recursion_limit:
            raise RecursionError('Maximum recursion depth exceeded: too many subqueries.')
    self.subq_aliases = self.subq_aliases.union([self.alias_prefix])
    other_query.subq_aliases = other_query.subq_aliases.union(self.subq_aliases)
    if exclude is None:
        exclude = {}
    self.change_aliases({alias: '%s%d' % (self.alias_prefix, pos) for pos, alias in enumerate(self.alias_map) if alias not in exclude})