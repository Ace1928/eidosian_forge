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
def typeof_storemap(self, inst):
    constraint = SetItemConstraint(target=inst.dct, index=inst.key, value=inst.value, loc=inst.loc)
    self.constraints.append(constraint)
    self.calls.append((inst, constraint))