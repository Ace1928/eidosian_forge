from collections import OrderedDict
from ... import sage_helper
@cached_method
def integral_homology_basis(self, dimension=1):
    C = self.chain_complex()
    homology = C.homology(generators=True)[dimension]
    ans = [factor[1].vector(dimension) for factor in homology]
    if dimension == 1:
        assert len(ans) == 2 - self.euler()
        ans = [OneCycle(self, a) for a in ans]
    return ans