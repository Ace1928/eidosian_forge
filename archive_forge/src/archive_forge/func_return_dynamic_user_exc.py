from collections import namedtuple
from collections.abc import Iterable
import itertools
import hashlib
from llvmlite import ir
from numba.core import types, cgutils, errors
from numba.core.base import PYOBJECT, GENERIC_POINTER
def return_dynamic_user_exc(self, builder, exc, exc_args, nb_types, loc=None, func_name=None):
    """
        Same as ::return_user_exc but for dynamic exceptions
        """
    self.set_dynamic_user_exc(builder, exc, exc_args, nb_types, loc=loc, func_name=func_name)
    self._return_errcode_raw(builder, RETCODE_USEREXC)