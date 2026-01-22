import abc
import copy
import types
import warnings
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from functools import lru_cache
import numpy as np
import pennylane as qml
from pennylane.measurements import (
from pennylane.operation import Observable, Operation, Tensor, Operator, StatePrepBase
from pennylane.ops import Hamiltonian, Sum
from pennylane.tape import QuantumScript, QuantumTape, expand_tape_state_prep
from pennylane.wires import WireError, Wires
from pennylane.queuing import QueuingManager
def execute_and_gradients(self, circuits, method='jacobian', **kwargs):
    """Execute a batch of quantum circuits on the device, and return both the
        results and the gradients.

        The circuits are represented by tapes, and they are executed
        one-by-one using the device's ``execute`` method. The results and the
        corresponding Jacobians are collected in a list.

        For plugin developers: This method should be overwritten if the device
        can efficiently run multiple circuits on a backend, for example using
        parallel and/or asynchronous executions, and return both the results and the
        Jacobians.

        Args:
            circuits (list[.tape.QuantumTape]): circuits to execute on the device
            method (str): the device method to call to compute the Jacobian of a single circuit
            **kwargs: keyword argument to pass when calling ``method``

        Returns:
            tuple[list[array[float]], list[array[float]]]: Tuple containing list of measured value(s)
            and list of Jacobians. Returned Jacobians should be of shape ``(output_shape, num_params)``.
        """
    if self.tracker.active:
        self.tracker.update(execute_and_derivative_batches=1, derivatives=len(circuits))
        self.tracker.record()
    gradient_method = getattr(self, method)
    res = []
    jacs = []
    for circuit in circuits:
        res.append(self.batch_execute([circuit])[0])
        jacs.append(gradient_method(circuit, **kwargs))
    return (res, jacs)