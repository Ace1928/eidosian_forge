import collections
import copy
import functools
import itertools
import operator
from heat.common import exception
from heat.engine import function
from heat.engine import properties
def update_policy_changed(self):
    """Return True if the resource update policy has changed."""
    return self.old_defn._update_policy != self.new_defn._update_policy