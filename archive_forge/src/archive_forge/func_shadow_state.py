import warnings
from functools import reduce, partial
from itertools import product
from typing import Sequence, Callable
import numpy as np
import pennylane as qml
from pennylane.measurements import ClassicalShadowMP
from pennylane.tape import QuantumScript, QuantumTape
from pennylane import transform
@partial(transform, final_transform=True)
def shadow_state(tape: QuantumTape, wires, diffable=False) -> (Sequence[QuantumTape], Callable):
    """Transform a circuit returning a classical shadow into one that returns
    the reconstructed state in a differentiable manner.

    Args:
        tape (QNode or QuantumTape or Callable): A quantum circuit.
        wires (list[int] or list[list[int]]): If a list of ints, this represents
            the wires over which to reconstruct the state. If a list of list of ints,
            a state is reconstructed for every element of the outer list, saving
            qfunc evaluations.
        diffable (bool): If True, reconstruct the state in a differentiable
            fashion, where the gradient of the reconstructed state approaches
            the gradient of the true state in expectation. This comes at a performance
            cost.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]:

        The transformed circuit as described in :func:`qml.transform <pennylane.transform>`. Executing this circuit
        will provide the reconstructed state in the form of a tensor.

    **Example**

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2, shots=10000)

        @partial(qml.shadows.shadow_state, wires=[0, 1], diffable=True)
        @qml.qnode(dev)
        def circuit(x):
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RX(x, wires=0)
            return qml.classical_shadow(wires=[0, 1])

    >>> x = np.array(1.2)
    >>> circuit(x)
    array([[ 0.33835   +0.j     , -0.01215   +0.2241j , -0.00465   +0.237j  ,  0.35504997-0.01755j],
       [-0.01215   -0.2241j ,  0.1528    +0.j     ,  0.16919999-0.0036j , -0.00285   -0.22065j],
       [-0.00465   -0.237j  ,  0.16919999+0.0036j ,  0.17529999+0.j     ,  0.0099    -0.2358j ],
       [ 0.35504997+0.01755j, -0.00285   +0.22065j,  0.0099    +0.2358j ,  0.33355   +0.j     ]], dtype=complex64)
    >>> qml.jacobian(lambda x: np.real(circuit(x)))(x)
    array([[-0.245025, -0.005325,  0.004275, -0.2358  ],
           [-0.005325,  0.235275,  0.2358  , -0.004275],
           [ 0.004275,  0.2358  ,  0.244875, -0.002175],
           [-0.2358  , -0.004275, -0.002175, -0.235125]])
    """
    tapes, fn = _shadow_state_diffable(tape, wires) if diffable else _shadow_state_undiffable(tape, wires)
    return (tapes, fn)