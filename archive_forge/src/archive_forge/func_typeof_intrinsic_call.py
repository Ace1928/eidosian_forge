import logging
import operator
import contextlib
import itertools
from pprint import pprint
from collections import OrderedDict, defaultdict
from functools import reduce
from numba.core import types, utils, typing, ir, config
from numba.core.typing.templates import Signature
from numba.core.errors import (TypingError, UntypedAttributeError,
from numba.core.funcdesc import qualifying_prefix
from numba.core.typeconv import Conversion
def typeof_intrinsic_call(self, inst, target, func, *args):
    constraint = IntrinsicCallConstraint(target.name, func, args, kws=(), vararg=None, loc=inst.loc)
    self.constraints.append(constraint)
    self.calls.append((inst.value, constraint))