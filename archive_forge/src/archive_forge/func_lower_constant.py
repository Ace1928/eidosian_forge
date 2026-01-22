import collections
import contextlib
import inspect
import functools
from enum import Enum
from numba.core import typing, types, utils, cgutils
from numba.core.typing.templates import BaseRegistryLoader
def lower_constant(self, ty):
    """
        Decorate the implementation for creating a constant of type *ty*.

        The decorated implementation will have the signature
        (context, builder, ty, pyval).
        """

    def decorate(impl):
        self.constants.append((impl, (ty,)))
        return impl
    return decorate