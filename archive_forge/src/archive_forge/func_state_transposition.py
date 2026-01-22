from functools import partial
from typing import List, Union, Sequence, Callable
import networkx as nx
import pennylane as qml
from pennylane.transforms import transform
from pennylane import Hamiltonian
from pennylane.operation import Tensor
from pennylane.ops import __all__ as all_ops
from pennylane.ops.qubit import SWAP
from pennylane.queuing import QueuingManager
from pennylane.tape import QuantumTape
def state_transposition(results, mps, new_wire_order, original_wire_order):
    """Transpose the order of any state return.

    Args:
        results (ResultBatch): the result of executing a batch of length 1

    Keyword Args:
        mps (List[MeasurementProcess]): A list of measurements processes. At least one is a ``StateMP``
        new_wire_order (Sequence[Any]): the wire order after transpile has been called
        original_wire_order (.Wires): the devices wire order

    Returns:
        Result: The result object with state dimensions transposed.

    """
    if len(mps) == 1:
        temp_mp = qml.measurements.StateMP(wires=original_wire_order)
        return temp_mp.process_state(results[0], wire_order=qml.wires.Wires(new_wire_order))
    new_results = list(results[0])
    for i, mp in enumerate(mps):
        if isinstance(mp, qml.measurements.StateMP):
            temp_mp = qml.measurements.StateMP(wires=original_wire_order)
            new_res = temp_mp.process_state(new_results[i], wire_order=qml.wires.Wires(new_wire_order))
            new_results[i] = new_res
    return tuple(new_results)