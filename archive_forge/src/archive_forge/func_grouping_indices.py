import itertools
import numbers
from collections.abc import Iterable
from copy import copy
import functools
from typing import List
import numpy as np
import scipy
import pennylane as qml
from pennylane.operation import Observable, Tensor
from pennylane.wires import Wires
@grouping_indices.setter
def grouping_indices(self, value):
    """Set the grouping indices, if known without explicit computation, or if
        computation was done externally. The groups are not verified.

        **Example**

        Examples of valid groupings for the Hamiltonian

        >>> H = qml.Hamiltonian([qml.X('a'), qml.X('b'), qml.Y('b')])

        are

        >>> H.grouping_indices = [[0, 1], [2]]

        or

        >>> H.grouping_indices = [[0, 2], [1]]

        since both ``qml.X('a'), qml.X('b')`` and ``qml.X('a'), qml.Y('b')`` commute.


        Args:
            value (list[list[int]]): List of lists of indexes of the observables in ``self.ops``. Each sublist
                represents a group of commuting observables.
        """
    if not isinstance(value, Iterable) or any((not isinstance(sublist, Iterable) for sublist in value)) or any((i not in range(len(self.ops)) for i in [i for sl in value for i in sl])):
        raise ValueError(f'The grouped index value needs to be a tuple of tuples of integers between 0 and the number of observables in the Hamiltonian; got {value}')
    self._grouping_indices = tuple((tuple(sublist) for sublist in value))