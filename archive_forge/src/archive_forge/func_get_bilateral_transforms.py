import itertools
import math
import warnings
from django.core.exceptions import EmptyResultSet, FullResultSet
from django.db.backends.base.operations import BaseDatabaseOperations
from django.db.models.expressions import Case, Expression, Func, Value, When
from django.db.models.fields import (
from django.db.models.query_utils import RegisterLookupMixin
from django.utils.datastructures import OrderedSet
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.functional import cached_property
from django.utils.hashable import make_hashable
def get_bilateral_transforms(self):
    if hasattr(self.lhs, 'get_bilateral_transforms'):
        bilateral_transforms = self.lhs.get_bilateral_transforms()
    else:
        bilateral_transforms = []
    if self.bilateral:
        bilateral_transforms.append(self.__class__)
    return bilateral_transforms