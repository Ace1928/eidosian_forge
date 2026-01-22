import numpy as np
import pennylane as qml
from pennylane.devices import DefaultQubitLegacy
from pennylane.pulse import ParametrizedEvolution
from pennylane.typing import TensorLike
@staticmethod
def states_to_binary(samples, num_wires, dtype=jnp.int32):
    """Convert basis states from base 10 to binary representation.

        This is an auxiliary method to the generate_samples method.

        Args:
            samples (List[int]): samples of basis states in base 10 representation
            num_wires (int): the number of qubits
            dtype (type): Type of the internal integer array to be used. Can be
                important to specify for large systems for memory allocation
                purposes.

        Returns:
            List[int]: basis states in binary representation
        """
    powers_of_two = 1 << jnp.arange(num_wires, dtype=dtype)
    states_sampled_base_ten = samples[..., None] & powers_of_two
    return (states_sampled_base_ten > 0).astype(dtype)[..., ::-1]