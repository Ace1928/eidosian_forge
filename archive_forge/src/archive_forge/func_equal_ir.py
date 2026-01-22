from collections import defaultdict
import copy
import itertools
import os
import linecache
import pprint
import re
import sys
import operator
from types import FunctionType, BuiltinFunctionType
from functools import total_ordering
from io import StringIO
from numba.core import errors, config
from numba.core.utils import (BINOPS_TO_OPERATORS, INPLACE_BINOPS_TO_OPERATORS,
from numba.core.errors import (NotDefinedError, RedefinedError,
from numba.core import consts
def equal_ir(self, other):
    """ Checks that the IR contained within is equal to the IR in other.
        Equality is defined by being equal in fundamental structure (blocks,
        labels, IR node type and the order in which they are defined) and the
        IR nodes being equal. IR node equality essentially comes down to
        ensuring a node's `.__dict__` or `.__slots__` is equal, with the
        exception of ignoring 'loc' and 'scope' entries. The upshot is that the
        comparison is essentially location and scope invariant, but otherwise
        behaves as unsurprisingly as possible.
        """
    if type(self) is type(other):
        return self.blocks == other.blocks
    return False