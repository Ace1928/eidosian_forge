from typing import TYPE_CHECKING
import numpy as np
from cirq.qis import to_valid_state_vector
def state_vector_to_probabilities(state_vector: 'cirq.STATE_VECTOR_LIKE') -> np.ndarray:
    """Function to transform a state vector like object into a numpy array of probabilities."""
    valid_state_vector = to_valid_state_vector(state_vector)
    return np.abs(valid_state_vector) ** 2