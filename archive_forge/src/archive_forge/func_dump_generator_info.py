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
def dump_generator_info(self, file=None):
    file = file or sys.stdout
    gi = self.generator_info
    print('generator state variables:', sorted(gi.state_vars), file=file)
    for index, yp in sorted(gi.yield_points.items()):
        print('yield point #%d: live variables = %s, weak live variables = %s' % (index, sorted(yp.live_vars), sorted(yp.weak_live_vars)), file=file)