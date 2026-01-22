from typing import List, Tuple
from numpy.typing import NDArray
import attr
import cirq
import numpy as np
from cirq._compat import cached_property
from cirq_ft import infra, linalg
from cirq_ft.algos import (
@classmethod
def from_lcu_probs(cls, lcu_probabilities: List[float], *, probability_epsilon: float=1e-05) -> 'StatePreparationAliasSampling':
    """Factory to construct the state preparation gate for a given set of LCU coefficients.

        Args:
            lcu_probabilities: The LCU coefficients.
            probability_epsilon: The desired accuracy to represent each probability
                (which sets mu size and keep/alt integers).
                See `cirq_ft.linalg.lcu_util.preprocess_lcu_coefficients_for_reversible_sampling`
                for more information.
        """
    alt, keep, mu = linalg.preprocess_lcu_coefficients_for_reversible_sampling(lcu_coefficients=lcu_probabilities, epsilon=probability_epsilon)
    N = len(lcu_probabilities)
    return StatePreparationAliasSampling(selection_registers=infra.SelectionRegister('selection', (N - 1).bit_length(), N), alt=np.array(alt), keep=np.array(keep), mu=mu)