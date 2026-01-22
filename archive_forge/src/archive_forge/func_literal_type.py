from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Dict as ptDict, Type as ptType
import itertools
import weakref
from functools import cached_property
import numpy as np
from numba.core.utils import get_hashable_key
@property
def literal_type(self):
    if self._literal_type_cache is None:
        from numba.core import typing
        ctx = typing.Context()
        try:
            res = ctx.resolve_value_type(self.literal_value)
        except ValueError as e:
            if 'Int value is too large' in str(e):
                msg = f'Cannot create literal type. {str(e)}'
                raise TypeError(msg)
            msg = "{} has no attribute 'literal_type'".format(self)
            raise AttributeError(msg)
        self._literal_type_cache = res
    return self._literal_type_cache