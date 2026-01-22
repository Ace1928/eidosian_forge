import copy
from threading import RLock
import pennylane as qml
from pennylane.measurements import CountsMP, ProbabilityMP, SampleMP, MeasurementProcess
from pennylane.operation import DecompositionUndefinedError, Operator, StatePrepBase
from pennylane.queuing import AnnotatedQueue, QueuingManager, process_queue
from pennylane.pytrees import register_pytree
from .qscript import QuantumScript
def expand_tape_state_prep(tape, skip_first=True):
    """Expand all instances of StatePrepBase operations in the tape.

    Args:
        tape (QuantumScript): The tape to expand.
        skip_first (bool): If ``True``, will not expand a ``StatePrepBase`` operation if
            it is the first operation in the tape.

    Returns:
        QuantumTape: The expanded version of ``tape``.

    **Example**

    If a ``StatePrepBase`` occurs as the first operation of a tape, the operation will not be expanded:

    >>> ops = [qml.StatePrep([0, 1], wires=0), qml.Z(1), qml.StatePrep([1, 0], wires=0)]
    >>> tape = qml.tape.QuantumScript(ops, [])
    >>> new_tape = qml.tape.tape.expand_tape_state_prep(tape)
    >>> new_tape.operations
    [StatePrep(array([0, 1]), wires=[0]), Z(1), MottonenStatePreparation(array([1, 0]), wires=[0])]

    To force expansion, the keyword argument ``skip_first`` can be set to ``False``:

    >>> new_tape = qml.tape.tape.expand_tape_state_prep(tape, skip_first=False)
    [MottonenStatePreparation(array([0, 1]), wires=[0]), Z(1), MottonenStatePreparation(array([1, 0]), wires=[0])]
    """
    first_op = tape.operations[0]
    new_ops = [first_op] if not isinstance(first_op, StatePrepBase) or skip_first else first_op.decomposition()
    for op in tape.operations[1:]:
        if isinstance(op, StatePrepBase):
            new_ops.extend(op.decomposition())
        else:
            new_ops.append(op)
    new_tape = tape.__class__(new_ops, tape.measurements, shots=tape.shots, _update=False)
    new_tape.wires = copy.copy(tape.wires)
    new_tape.num_wires = tape.num_wires
    new_tape._batch_size = tape._batch_size
    new_tape._output_dim = tape._output_dim
    return new_tape