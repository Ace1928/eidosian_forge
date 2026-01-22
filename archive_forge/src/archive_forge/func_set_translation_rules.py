import collections
import copy
import functools
import itertools
import operator
from heat.common import exception
from heat.engine import function
from heat.engine import properties
def set_translation_rules(self, rules=None, client_resolve=True):
    """Helper method to update properties with translation rules."""
    self._rules = rules or []
    self._client_resolve = client_resolve