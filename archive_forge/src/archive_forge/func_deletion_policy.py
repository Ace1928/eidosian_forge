import collections
import copy
import functools
import itertools
import operator
from heat.common import exception
from heat.engine import function
from heat.engine import properties
def deletion_policy(self):
    """Return the deletion policy for the resource.

        The policy will be one of those listed in DELETION_POLICIES.
        """
    return function.resolve(self._deletion_policy) or self.DELETE