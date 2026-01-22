import copy
import re
from functools import total_ordering
from nltk.internals import raise_unorderable_types, read_str
from nltk.sem.logic import (
def read_app_value(self, s, position, reentrances, match):
    """Mainly included for backwards compat."""
    return (self._logic_parser.parse('%s(%s)' % match.group(2, 3)), match.end())