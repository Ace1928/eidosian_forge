from cmath import inf
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy import Matrix, pi
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import zeros
from sympy import sin, cos
def remove_member(self, label):
    """
        This method removes a member from the given truss.

        Parameters
        ==========
        label: String or Symbol
            The label for the member to be removed.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> t = Truss()
        >>> t.add_node('A', 0, 0)
        >>> t.add_node('B', 3, 0)
        >>> t.add_node('C', 2, 2)
        >>> t.add_member('AB', 'A', 'B')
        >>> t.add_member('AC', 'A', 'C')
        >>> t.add_member('BC', 'B', 'C')
        >>> t.members
        {'AB': ['A', 'B'], 'AC': ['A', 'C'], 'BC': ['B', 'C']}
        >>> t.remove_member('AC')
        >>> t.members
        {'AB': ['A', 'B'], 'BC': ['B', 'C']}
        """
    if label not in list(self._members):
        raise ValueError('No such member exists in the Truss')
    else:
        self._nodes_occupied.pop((self._members[label][0], self._members[label][1]))
        self._nodes_occupied.pop((self._members[label][1], self._members[label][0]))
        self._members.pop(label)
        self._internal_forces.pop(label)