from typing import Callable, Mapping, Optional, Sequence
import numpy as np
from cirq.circuits import Circuit
from cirq.ops import QubitOrder, QubitOrderOrList
from cirq.sim import final_state_vector
from cirq.value import state_vector_to_probabilities
def hog_score_xeb_fidelity_from_probabilities(hilbert_space_dimension: int, probabilities: Sequence[float]) -> float:
    """XEB fidelity estimator based on normalized HOG score.

    Estimates fidelity from ideal probabilities of observed bitstrings.

    See `linear_xeb_fidelity_from_probabilities` for the assumptions made
    by this estimator.

    The mean of this estimator is the true fidelity f and the variance is

        (1/log(2)^2 - f^2) / M

    where f is the fidelity and M the number of observations, equal to
    len(probabilities). This is always worse than log XEB (see above).
    Since this estimator is unbiased, the variance is equal to the mean
    squared error of the estimator.

    The estimator is intended for use with xeb_fidelity() below. It is
    based on the HOG problem defined in https://arxiv.org/abs/1612.05903.

    Args:
        hilbert_space_dimension: Dimension of the Hilbert space on which
           the channel whose fidelity is being estimated is defined.
        probabilities: Ideal probabilities of bitstrings observed in
            experiment.
    Returns:
        Estimate of fidelity associated with an experimental realization
        of a quantum circuit.
    """
    score = np.mean(probabilities > np.log(2) / hilbert_space_dimension)
    return (2 * score - 1) / np.log(2)