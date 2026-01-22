from .cartan_type import CartanType
from mpmath import fac
from sympy.core.backend import Matrix, eye, Rational, igcd
from sympy.core.basic import Atom

        This method returns the Coxeter diagram corresponding to a Weyl group.
        The Coxeter diagram can be obtained from a Lie algebra's Dynkin diagram
        by deleting all arrows; the Coxeter diagram is the undirected graph.
        The vertices of the Coxeter diagram represent the generating reflections
        of the Weyl group, $s_i$.  An edge is drawn between $s_i$ and $s_j$ if the order
        $m(i, j)$ of $s_is_j$ is greater than two.  If there is one edge, the order
        $m(i, j)$ is 3.  If there are two edges, the order $m(i, j)$ is 4, and if there
        are three edges, the order $m(i, j)$ is 6.

        Examples
        ========

        >>> from sympy.liealgebras.weyl_group import WeylGroup
        >>> c = WeylGroup("B3")
        >>> print(c.coxeter_diagram())
        0---0===0
        1   2   3
        