from typing import List, Tuple
from numpy.typing import NDArray
import attr
import cirq
import numpy as np
from cirq._compat import cached_property
from cirq_ft import infra, linalg
from cirq_ft.algos import (
Factory to construct the state preparation gate for a given set of LCU coefficients.

        Args:
            lcu_probabilities: The LCU coefficients.
            probability_epsilon: The desired accuracy to represent each probability
                (which sets mu size and keep/alt integers).
                See `cirq_ft.linalg.lcu_util.preprocess_lcu_coefficients_for_reversible_sampling`
                for more information.
        