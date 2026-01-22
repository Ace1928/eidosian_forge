import contextlib
import pennylane as qml
from pennylane.operation import (
Context manager for setting custom decompositions.

    Args:
        custom_decomps (Dict[Union(str, qml.operation.Operation), Callable]): Custom
            decompositions to be applied by the device at runtime.
        dev (pennylane.Device): A quantum device.
        decomp_depth: The maximum depth of the expansion.

    **Example**

    Suppose we would like a custom expansion function that decomposes all CNOTs
    into CZs. We first define a decomposition function:

    .. code-block:: python

        def custom_cnot(wires):
            return [
                qml.Hadamard(wires=wires[1]),
                qml.CZ(wires=[wires[0], wires[1]]),
                qml.Hadamard(wires=wires[1])
            ]

    This context manager can be used to temporarily change a devices expansion
    function to one that takes into account the custom decompositions.

    .. code-block:: python

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, expansion_strategy="device")
        def circuit():
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Z(0))

    >>> print(qml.draw(circuit)())
    0: ─╭●─┤  <Z>
    1: ─╰X─┤

    Now let's set up a context where the custom decomposition will be applied:

    >>> with qml.transforms.set_decomposition({qml.CNOT : custom_cnot}, dev):
    ...     print(qml.draw(circuit, wire_order=[0, 1])())
    0: ────╭●────┤  <Z>
    1: ──H─╰Z──H─┤

    