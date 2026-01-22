from .abstract import ArrayCompatible, Dummy, IterableType, IteratorType
from numba.core.errors import NumbaTypeError, NumbaValueError
@property
def is_f_contig(self):
    return self.layout == 'F' or (self.ndim <= 1 and self.layout in 'CF')