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
def unify_finished(self, typdict, retty, fntys):
    print('Variable types'.center(80, '-'))
    pprint(typdict)
    print('Return type'.center(80, '-'))
    pprint(retty)
    print('Call types'.center(80, '-'))
    pprint(fntys)