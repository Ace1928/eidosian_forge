from typing import Sequence, Callable
from pennylane.ops.op_math import Adjoint
from pennylane.wires import Wires
from pennylane.tape import QuantumTape
from pennylane.transforms import transform
from pennylane.ops.qubit.attributes import (
from .optimization_utils import find_next_gate
@transform
def cancel_inverses(tape: QuantumTape) -> (Sequence[QuantumTape], Callable):
    """Quantum function transform to remove any operations that are applied next to their
    (self-)inverses or adjoint.

    Args:
        tape (QNode or QuantumTape or Callable): A quantum circuit.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]: The transformed circuit as described in :func:`qml.transform <pennylane.transform>`.


    **Example**

    You can apply the cancel inverses transform directly on :class:`~.QNode`.

    >>> dev = qml.device('default.qubit', wires=3)

    .. code-block:: python

        @cancel_inverses
        @qml.qnode(device=dev)
        def circuit(x, y, z):
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=1)
            qml.Hadamard(wires=0)
            qml.RX(x, wires=2)
            qml.RY(y, wires=1)
            qml.X(1)
            qml.RZ(z, wires=0)
            qml.RX(y, wires=2)
            qml.CNOT(wires=[0, 2])
            qml.X(1)
            return qml.expval(qml.Z(0))

    >>> circuit(0.1, 0.2, 0.3)
    0.999999999999999

    .. details::
        :title: Usage Details

        You can also apply it on quantum functions:

        .. code-block:: python

            def qfunc(x, y, z):
                qml.Hadamard(wires=0)
                qml.Hadamard(wires=1)
                qml.Hadamard(wires=0)
                qml.RX(x, wires=2)
                qml.RY(y, wires=1)
                qml.X(1)
                qml.RZ(z, wires=0)
                qml.RX(y, wires=2)
                qml.CNOT(wires=[0, 2])
                qml.X(1)
                return qml.expval(qml.Z(0))

        The circuit before optimization:

        >>> qnode = qml.QNode(qfunc, dev)
        >>> print(qml.draw(qnode)(1, 2, 3))
        0: ──H─────────H─────────RZ(3.00)─╭●────┤  <Z>
        1: ──H─────────RY(2.00)──X────────│───X─┤
        2: ──RX(1.00)──RX(2.00)───────────╰X────┤

        We can see that there are two adjacent Hadamards on the first qubit that
        should cancel each other out. Similarly, there are two Pauli-X gates on the
        second qubit that should cancel. We can obtain a simplified circuit by running
        the ``cancel_inverses`` transform:

        >>> optimized_qfunc = cancel_inverses(qfunc)
        >>> optimized_qnode = qml.QNode(optimized_qfunc, dev)
        >>> print(qml.draw(optimized_qnode)(1, 2, 3))
        0: ──RZ(3.00)───────────╭●─┤  <Z>
        1: ──H─────────RY(2.00)─│──┤
        2: ──RX(1.00)──RX(2.00)─╰X─┤

    """
    list_copy = tape.operations.copy()
    operations = []
    while len(list_copy) > 0:
        current_gate = list_copy[0]
        list_copy.pop(0)
        next_gate_idx = find_next_gate(current_gate.wires, list_copy)
        if next_gate_idx is None:
            operations.append(current_gate)
            continue
        next_gate = list_copy[next_gate_idx]
        if _are_inverses(current_gate, next_gate):
            if current_gate.wires == next_gate.wires:
                list_copy.pop(next_gate_idx)
                continue
            if len(Wires.shared_wires([current_gate.wires, next_gate.wires])) != len(current_gate.wires):
                operations.append(current_gate)
                continue
            if current_gate in symmetric_over_all_wires:
                list_copy.pop(next_gate_idx)
                continue
            if current_gate in symmetric_over_control_wires:
                if len(Wires.shared_wires([current_gate.wires[:-1], next_gate.wires[:-1]])) == len(current_gate.wires) - 1:
                    list_copy.pop(next_gate_idx)
                    continue
        operations.append(current_gate)
        continue
    new_tape = type(tape)(operations, tape.measurements, shots=tape.shots)

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]
    return ([new_tape], null_postprocessing)