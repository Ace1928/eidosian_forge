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
def solve_lookup_type(self, lookup, summarize=False):
    """
        Solve the lookup type from the lookup (e.g.: 'foobar__id__icontains').
        """
    lookup_splitted = lookup.split(LOOKUP_SEP)
    if self.annotations:
        annotation, expression_lookups = refs_expression(lookup_splitted, self.annotations)
        if annotation:
            expression = self.annotations[annotation]
            if summarize:
                expression = Ref(annotation, expression)
            return (expression_lookups, (), expression)
    _, field, _, lookup_parts = self.names_to_path(lookup_splitted, self.get_meta())
    field_parts = lookup_splitted[0:len(lookup_splitted) - len(lookup_parts)]
    if len(lookup_parts) > 1 and (not field_parts):
        raise FieldError('Invalid lookup "%s" for model %s".' % (lookup, self.get_meta().model.__name__))
    return (lookup_parts, field_parts, False)