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
@property
def versioned_names(self):
    """Known versioned names for this variable, i.e. known variable names in
        the scope that have been formed from applying SSA to this variable
        """
    return self.scope.get_versions_of(self.unversioned_name)