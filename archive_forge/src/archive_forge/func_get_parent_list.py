import bisect
import copy
import inspect
import warnings
from collections import defaultdict
from django.apps import apps
from django.conf import settings
from django.core.exceptions import FieldDoesNotExist, ImproperlyConfigured
from django.db import connections
from django.db.models import AutoField, Manager, OrderWrt, UniqueConstraint
from django.db.models.query_utils import PathInfo
from django.utils.datastructures import ImmutableList, OrderedSet
from django.utils.deprecation import RemovedInDjango51Warning
from django.utils.functional import cached_property
from django.utils.module_loading import import_string
from django.utils.text import camel_case_to_spaces, format_lazy
from django.utils.translation import override
def get_parent_list(self):
    """
        Return all the ancestors of this model as a list ordered by MRO.
        Useful for determining if something is an ancestor, regardless of lineage.
        """
    result = OrderedSet(self.parents)
    for parent in self.parents:
        for ancestor in parent._meta.get_parent_list():
            result.add(ancestor)
    return list(result)