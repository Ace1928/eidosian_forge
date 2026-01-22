from copy import copy
import pennylane as qml
from pennylane.operation import Operator
from pennylane.ops import Sum
from pennylane.ops.qubit.hamiltonian import Hamiltonian
from pennylane.typing import TensorLike
from pennylane.wires import Wires
The addition operation between a ``ParametrizedHamiltonian`` and an ``Operator``
        or ``ParametrizedHamiltonian``.