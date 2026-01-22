from collections import Counter
from typing import Optional, Sequence
import warnings
from numpy.random import default_rng
import numpy as np
import pennylane as qml
from pennylane.measurements import (
from pennylane.typing import Result
from .initialize_state import create_initial_state
from .apply_operation import apply_operation
from .measure import measure
from .sampling import measure_with_samples
def get_final_state(circuit, debugger=None, interface=None, mid_measurements=None):
    """
    Get the final state that results from executing the given quantum script.

    This is an internal function that will be called by the successor to ``default.qubit``.

    Args:
        circuit (.QuantumScript): The single circuit to simulate
        debugger (._Debugger): The debugger to use
        interface (str): The machine learning interface to create the initial state with
        mid_measurements (None, dict): Dictionary of mid-circuit measurements

    Returns:
        Tuple[TensorLike, bool]: A tuple containing the final state of the quantum script and
            whether the state has a batch dimension.

    """
    circuit = circuit.map_to_standard_wires()
    prep = None
    if len(circuit) > 0 and isinstance(circuit[0], qml.operation.StatePrepBase):
        prep = circuit[0]
    state = create_initial_state(sorted(circuit.op_wires), prep, like=INTERFACE_TO_LIKE[interface])
    is_state_batched = bool(prep and prep.batch_size is not None)
    for op in circuit.operations[bool(prep):]:
        state = apply_operation(op, state, is_state_batched=is_state_batched, debugger=debugger, mid_measurements=mid_measurements)
        if isinstance(op, qml.Projector):
            state, circuit._shots = _postselection_postprocess(state, is_state_batched, circuit.shots)
        is_state_batched = is_state_batched or op.batch_size is not None
    for _ in range(len(circuit.wires) - len(circuit.op_wires)):
        state = qml.math.stack([state, qml.math.zeros_like(state)], axis=-1)
    return (state, is_state_batched)