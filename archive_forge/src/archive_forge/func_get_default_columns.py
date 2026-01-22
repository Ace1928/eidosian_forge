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
def get_default_columns(self, select_mask, start_alias=None, opts=None, from_parent=None):
    """
        Compute the default columns for selecting every field in the base
        model. Will sometimes be called to pull in related models (e.g. via
        select_related), in which case "opts" and "start_alias" will be given
        to provide a starting point for the traversal.

        Return a list of strings, quoted appropriately for use in SQL
        directly, as well as a set of aliases used in the select statement (if
        'as_pairs' is True, return a list of (alias, col_name) pairs instead
        of strings as the first component and None as the second component).
        """
    result = []
    if opts is None:
        if (opts := self.query.get_meta()) is None:
            return result
    start_alias = start_alias or self.query.get_initial_alias()
    seen_models = {None: start_alias}
    for field in opts.concrete_fields:
        model = field.model._meta.concrete_model
        if model == opts.model:
            model = None
        if from_parent and model is not None and issubclass(from_parent._meta.concrete_model, model._meta.concrete_model):
            continue
        if select_mask and field not in select_mask:
            continue
        alias = self.query.join_parent_model(opts, model, start_alias, seen_models)
        column = field.get_col(alias)
        result.append(column)
    return result