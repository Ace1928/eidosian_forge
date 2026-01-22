from .abstract import ArrayCompatible, Dummy, IterableType, IteratorType
from numba.core.errors import NumbaTypeError, NumbaValueError
@property
def is_c_contig(self):
    return self.layout == 'C' or (self.ndim <= 1 and self.layout in 'CF')