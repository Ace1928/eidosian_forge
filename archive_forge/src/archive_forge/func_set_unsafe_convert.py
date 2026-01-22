from numba.core.typeconv import castgraph, Conversion
from numba.core import types
def set_unsafe_convert(self, fromty, toty):
    self.set_compatible(fromty, toty, Conversion.unsafe)