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
def parse_native_mid_circuit_measurements(circuit: qml.tape.QuantumScript, all_shot_meas, mcm_shot_meas):
    """Combines, gathers and normalizes the results of native mid-circuit measurement runs.

    Args:
        circuit (QuantumTape): A one-shot (auxiliary) ``QuantumScript``
        all_shot_meas (Sequence[Any]): List of accumulated measurement results
        mcm_shot_meas (Sequence[dict]): List of dictionaries containing the mid-circuit measurement results of each shot

    Returns:
        tuple(TensorLike): The results of the simulation
    """

    def measurement_with_no_shots(measurement):
        return np.nan * np.ones_like(measurement.eigvals()) if isinstance(measurement, ProbabilityMP) else np.nan
    normalized_meas = []
    for i, m in enumerate(circuit.measurements):
        if not isinstance(m, (CountsMP, ExpectationMP, ProbabilityMP, SampleMP, VarianceMP)):
            raise ValueError(f'Native mid-circuit measurement mode does not support {type(m).__name__} measurements.')
        if m.mv and (not mcm_shot_meas):
            meas = measurement_with_no_shots(m)
        elif m.mv:
            meas = gather_mcm(m, mcm_shot_meas)
        elif not all_shot_meas:
            meas = measurement_with_no_shots(m)
        else:
            meas = gather_non_mcm(m, all_shot_meas[i], mcm_shot_meas)
        if isinstance(m, SampleMP):
            meas = qml.math.squeeze(meas)
        normalized_meas.append(meas)
    return tuple(normalized_meas) if len(normalized_meas) > 1 else normalized_meas[0]