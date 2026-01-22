from cmath import inf
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy import Matrix, pi
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import zeros
from sympy import sin, cos

        This method solves for all reaction forces of all supports and all internal forces
        of all the members in the truss, provided the Truss is solvable.

        A Truss is solvable if the following condition is met,

        2n >= r + m

        Where n is the number of nodes, r is the number of reaction forces, where each pinned
        support has 2 reaction forces and each roller has 1, and m is the number of members.

        The given condition is derived from the fact that a system of equations is solvable
        only when the number of variables is lesser than or equal to the number of equations.
        Equilibrium Equations in x and y directions give two equations per node giving 2n number
        equations. However, the truss needs to be stable as well and may be unstable if 2n > r + m.
        The number of variables is simply the sum of the number of reaction forces and member
        forces.

        .. note::
           The sign convention for the internal forces present in a member revolves around whether each
           force is compressive or tensile. While forming equations for each node, internal force due
           to a member on the node is assumed to be away from the node i.e. each force is assumed to
           be compressive by default. Hence, a positive value for an internal force implies the
           presence of compressive force in the member and a negative value implies a tensile force.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> t = Truss()
        >>> t.add_node("node_1", 0, 0)
        >>> t.add_node("node_2", 6, 0)
        >>> t.add_node("node_3", 2, 2)
        >>> t.add_node("node_4", 2, 0)
        >>> t.add_member("member_1", "node_1", "node_4")
        >>> t.add_member("member_2", "node_2", "node_4")
        >>> t.add_member("member_3", "node_1", "node_3")
        >>> t.add_member("member_4", "node_2", "node_3")
        >>> t.add_member("member_5", "node_3", "node_4")
        >>> t.apply_load("node_4", magnitude=10, direction=270)
        >>> t.apply_support("node_1", type="pinned")
        >>> t.apply_support("node_2", type="roller")
        >>> t.solve()
        >>> t.reaction_loads
        {'R_node_1_x': 0, 'R_node_1_y': 20/3, 'R_node_2_y': 10/3}
        >>> t.internal_forces
        {'member_1': 20/3, 'member_2': 20/3, 'member_3': -20*sqrt(2)/3, 'member_4': -10*sqrt(5)/3, 'member_5': 10}
        