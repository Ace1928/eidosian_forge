import functools
import inspect
import logging
from collections import namedtuple
from django.core.exceptions import FieldError
from django.db import DEFAULT_DB_ALIAS, DatabaseError, connections
from django.db.models.constants import LOOKUP_SEP
from django.utils import tree
from django.utils.functional import cached_property
from django.utils.hashable import make_hashable
def register_class_lookup(cls, lookup, lookup_name=None):
    if lookup_name is None:
        lookup_name = lookup.lookup_name
    if 'class_lookups' not in cls.__dict__:
        cls.class_lookups = {}
    cls.class_lookups[lookup_name] = lookup
    cls._clear_cached_class_lookups()
    return lookup