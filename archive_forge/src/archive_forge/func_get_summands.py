import warnings
import itertools
from copy import copy
from typing import List
import pennylane as qml
from pennylane import math
from pennylane.operation import Operator, convert_to_opmath
from pennylane.ops.qubit import Hamiltonian
from pennylane.queuing import QueuingManager
from .composite import CompositeOp
def get_summands(self, cutoff=1e-12):
    """Get summands list.

        All summands with a coefficient less than cutoff are ignored.

        Args:
            cutoff (float, optional): Cutoff value. Defaults to 1.0e-12.
        """
    new_summands = []
    for coeff, summand in self.queue.values():
        if coeff == 1:
            new_summands.append(summand)
        elif abs(coeff) > cutoff:
            new_summands.append(qml.s_prod(coeff, summand))
    return new_summands