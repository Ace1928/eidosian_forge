from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
def induced_representation(self, N):
    """
        Given a PSL(2,C) representation constructs the induced representation
        for the given N.
        The induced representation is in SL(N,C) if N is odd and
        SL(N,C) / {+1,-1} if N is even and is described in the Introduction of
        Garoufalidis, Thurston, Zickert
        The Complex Volume of SL(n,C)-Representations of 3-Manifolds
        http://arxiv.org/abs/1111.2828

        There is a canonical group homomorphism SL(2,C)->SL(N,C) coming from
        the the natural SL(2,C)-action on the vector space Sym^{N-1}(C^2).
        This homomorphisms decends to a homomorphism from PSL(2,C) if one
        divides the right side by {+1,-1} when N is even.
        Composing a representation with this homomorphism gives the induced
        representation.
        """
    num_tetrahedra = self.num_tetrahedra()
    if self.N() != 2:
        raise Exception('Cross ratios need to come from a PSL(2,C) representation')

    def key_value_pair(v, t, index):
        new_key = v + '_%d%d%d%d' % tuple(index) + '_%d' % t
        old_key = v + '_0000' + '_%d' % t
        return (new_key, self[old_key])
    d = dict([key_value_pair(v, t, index) for v in ['z', 'zp', 'zpp'] for t in range(num_tetrahedra) for index in utilities.quadruples_with_fixed_sum_iterator(N - 2)])
    return CrossRatios(d, is_numerical=self._is_numerical, manifold_thunk=self._manifold_thunk)