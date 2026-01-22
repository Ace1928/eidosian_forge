from numba.core.typeconv import castgraph, Conversion
from numba.core import types
def promote_unsafe(self, a, b):
    """
        Set `a` can promote to `b` and `b` can unsafe convert to `a`
        """
    self.promote(a, b)
    self.unsafe(b, a)