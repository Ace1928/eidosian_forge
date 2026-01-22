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
def get_extra_select(self, order_by, select):
    extra_select = []
    if self.query.distinct and (not self.query.distinct_fields):
        select_sql = [t[1] for t in select]
        for expr, (sql, params, is_ref) in order_by:
            without_ordering = self.ordering_parts.search(sql)[1]
            if not is_ref and (without_ordering, params) not in select_sql:
                extra_select.append((expr, (without_ordering, params), None))
    return extra_select