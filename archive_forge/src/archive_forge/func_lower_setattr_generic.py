import collections
import contextlib
import inspect
import functools
from enum import Enum
from numba.core import typing, types, utils, cgutils
from numba.core.typing.templates import BaseRegistryLoader
def lower_setattr_generic(self, ty):
    """
        Decorate the fallback implementation of __setattr__ for type *ty*.

        The decorated implementation will have the signature
        (context, builder, sig, args, attr).  The implementation is
        called for attributes which haven't been explicitly registered
        with lower_setattr().
        """
    return self.lower_setattr(ty, None)