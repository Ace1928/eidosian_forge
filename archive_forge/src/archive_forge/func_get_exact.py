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
def get_exact(self, name):
    """
        Refer to a variable.  The returned variable has the exact
        name (exact variable version).
        """
    try:
        return self.localvars.get(name)
    except NotDefinedError:
        if self.has_parent:
            return self.parent.get(name)
        else:
            raise