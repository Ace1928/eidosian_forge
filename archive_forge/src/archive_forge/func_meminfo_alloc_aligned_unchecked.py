import functools
from collections import namedtuple
from llvmlite import ir
from numba.core import types, cgutils, errors, config
from numba.core.utils import PYVERSION
def meminfo_alloc_aligned_unchecked(self, builder, size, align):
    """
        Allocate a new MemInfo with an aligned data payload of `size` bytes.
        The data pointer is aligned to `align` bytes.  `align` can be either
        a Python int or a LLVM uint32 value.

        A pointer to the MemInfo is returned.

        Returns NULL to indicate error/failure to allocate.
        """
    self._require_nrt()
    mod = builder.module
    u32 = ir.IntType(32)
    fnty = ir.FunctionType(cgutils.voidptr_t, [cgutils.intp_t, u32])
    fn = cgutils.get_or_insert_function(mod, fnty, self._meminfo_api.alloc_aligned)
    fn.return_value.add_attribute('noalias')
    if isinstance(align, int):
        align = self._context.get_constant(types.uint32, align)
    else:
        assert align.type == u32, 'align must be a uint32'
    return builder.call(fn, [size, align])