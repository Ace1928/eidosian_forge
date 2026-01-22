from .cartan_type import CartanType
from sympy.core.basic import Atom
def root_space(self):
    """Return the span of the simple roots

        The root space is the vector space spanned by the simple roots, i.e. it
        is a vector space with a distinguished basis, the simple roots.  This
        method returns a string that represents the root space as the span of
        the simple roots, alpha[1],...., alpha[n].

        Examples
        ========

        >>> from sympy.liealgebras.root_system import RootSystem
        >>> c = RootSystem("A3")
        >>> c.root_space()
        'alpha[1] + alpha[2] + alpha[3]'

        """
    n = self.cartan_type.rank()
    rs = ' + '.join(('alpha[' + str(i) + ']' for i in range(1, n + 1)))
    return rs