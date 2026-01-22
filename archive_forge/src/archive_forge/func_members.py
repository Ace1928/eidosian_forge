import collections
import warnings
from functools import cached_property
from llvmlite import ir
from .abstract import DTypeSpec, IteratorType, MutableSequence, Number, Type
from .common import Buffer, Opaque, SimpleIteratorType
from numba.core.typeconv import Conversion
from numba.core import utils
from .misc import UnicodeType
from .containers import Bytes
import numpy as np
@property
def members(self):
    """An ordered list of (name, type) for the fields.
        """
    ordered = sorted(self.fields.items(), key=lambda x: x[1].offset)
    return [(k, v.type) for k, v in ordered]