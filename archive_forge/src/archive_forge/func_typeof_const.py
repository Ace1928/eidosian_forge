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
def typeof_const(self, inst, target, const):
    ty = self.resolve_value_type(inst, const)
    if inst.value.use_literal_type:
        lit = types.maybe_literal(value=const)
    else:
        lit = None
    self.add_type(target.name, lit or ty, loc=inst.loc)