import collections.abc
import inspect
import warnings
from math import ceil
from django.utils.functional import cached_property
from django.utils.inspect import method_has_no_args
from django.utils.translation import gettext_lazy as _
def end_index(self):
    """
        Return the 1-based index of the last object on this page,
        relative to total objects found (hits).
        """
    if self.number == self.paginator.num_pages:
        return self.paginator.count
    return self.number * self.paginator.per_page