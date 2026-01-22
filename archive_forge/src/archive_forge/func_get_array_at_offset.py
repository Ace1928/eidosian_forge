from collections import namedtuple
import numpy as np
from llvmlite.ir import Constant, IRBuilder
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.compiler_lock import global_compiler_lock
from numba.core.caching import make_library_cache, NullCache
def get_array_at_offset(self, ind):
    return self._loader.load(context=self.context, builder=self.builder, data=self.data, ind=ind)