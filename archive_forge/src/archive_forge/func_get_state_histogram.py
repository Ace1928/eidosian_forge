from typing import cast, Optional, Sequence, SupportsFloat, Union
import collections
import numpy as np
import matplotlib.pyplot as plt
import cirq.study.result as result
def get_state_histogram(result: 'result.Result') -> np.ndarray:
    """Computes a state histogram from a single result with repetitions.

    Args:
        result: The trial result containing measurement results from which the
                state histogram should be computed.

    Returns:
        The state histogram (a numpy array) corresponding to the trial result.
    """
    num_qubits = sum([value.shape[1] for value in result.measurements.values()])
    states = 2 ** num_qubits
    values = np.zeros(states)
    measurement_by_result = np.hstack(list(result.measurements.values()))
    for meas in measurement_by_result:
        state_ind = int(''.join([str(x) for x in [int(x) for x in meas]]), 2)
        values[state_ind] += 1
    return values