from typing import List, Union, Tuple
import numpy as np
import pennylane as qml
from pennylane.ops import Sum, Hamiltonian, SProd, Prod
from pennylane.measurements import (
from pennylane.typing import TensorLike
from .apply_operation import apply_operation
from .measure import flatten_state
def get_num_shots_and_executions(tape: qml.tape.QuantumTape) -> Tuple[int, int]:
    """Get the total number of qpu executions and shots.

    Args:
        tape (qml.tape.QuantumTape): the tape we want to get the number of executions and shots for

    Returns:
        int, int: the total number of QPU executions and the total number of shots

    """
    groups, _ = _group_measurements(tape.measurements)
    num_executions = 0
    num_shots = 0
    for group in groups:
        if isinstance(group[0], ExpectationMP) and isinstance(group[0].obs, qml.Hamiltonian):
            indices = group[0].obs.grouping_indices
            H_executions = len(indices) if indices else len(group[0].obs.ops)
            num_executions += H_executions
            if tape.shots:
                num_shots += tape.shots.total_shots * H_executions
        elif isinstance(group[0], ExpectationMP) and isinstance(group[0].obs, qml.ops.Sum):
            num_executions += len(group[0].obs)
            if tape.shots:
                num_shots += tape.shots.total_shots * len(group[0].obs)
        elif isinstance(group[0], (ClassicalShadowMP, ShadowExpvalMP)):
            num_executions += tape.shots.total_shots
            if tape.shots:
                num_shots += tape.shots.total_shots
        else:
            num_executions += 1
            if tape.shots:
                num_shots += tape.shots.total_shots
    if tape.batch_size:
        num_executions *= tape.batch_size
        if tape.shots:
            num_shots *= tape.batch_size
    return (num_executions, num_shots)