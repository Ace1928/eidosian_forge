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
def try_transform(self, lhs, name):
    """
        Helper method for build_lookup(). Try to fetch and initialize
        a transform for name parameter from lhs.
        """
    transform_class = lhs.get_transform(name)
    if transform_class:
        return transform_class(lhs)
    else:
        output_field = lhs.output_field.__class__
        suggested_lookups = difflib.get_close_matches(name, lhs.output_field.get_lookups())
        if suggested_lookups:
            suggestion = ', perhaps you meant %s?' % ' or '.join(suggested_lookups)
        else:
            suggestion = '.'
        raise FieldError("Unsupported lookup '%s' for %s or join on the field not permitted%s" % (name, output_field.__name__, suggestion))