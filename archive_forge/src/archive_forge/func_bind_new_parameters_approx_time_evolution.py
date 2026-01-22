from typing import Sequence, Union
import copy
from functools import singledispatch
import pennylane as qml
from pennylane.typing import TensorLike
from pennylane.operation import Operator, Tensor
from ..identity import Identity
from ..qubit import Projector
from ..op_math import CompositeOp, SymbolicOp, ScalarSymbolicOp, Adjoint, Pow, SProd
@bind_new_parameters.register
def bind_new_parameters_approx_time_evolution(op: qml.ApproxTimeEvolution, params: Sequence[TensorLike]):
    new_hamiltonian = bind_new_parameters(op.hyperparameters['hamiltonian'], params[:-1])
    time = params[-1]
    n = op.hyperparameters['n']
    return qml.ApproxTimeEvolution(new_hamiltonian, time, n)