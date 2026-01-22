import collections
import copy
import functools
import itertools
import operator
from heat.common import exception
from heat.engine import function
from heat.engine import properties
def rawattrs():
    """Get an attribute with function objects stripped out."""
    for key, attr in attrs.items():
        value = getattr(self, attr)
        if value is not None:
            yield (key, copy.deepcopy(value))