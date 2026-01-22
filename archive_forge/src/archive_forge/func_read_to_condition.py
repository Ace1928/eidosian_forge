import inspect
import textwrap
import re
import pydoc
from warnings import warn
from collections import namedtuple
from collections.abc import Callable, Mapping
import copy
import sys
def read_to_condition(self, condition_func):
    start = self._l
    for line in self[start:]:
        if condition_func(line):
            return self[start:self._l]
        self._l += 1
        if self.eof():
            return self[start:self._l + 1]
    return []