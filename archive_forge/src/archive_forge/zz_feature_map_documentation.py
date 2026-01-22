from typing import Callable, List, Union, Optional
import numpy as np
from .pauli_feature_map import PauliFeatureMap
Create a new second-order Pauli-Z expansion.

        Args:
            feature_dimension: Number of features.
            reps: The number of repeated circuits, has a min. value of 1.
            entanglement: Specifies the entanglement structure. Refer to
                :class:`~qiskit.circuit.library.NLocal` for detail.
            data_map_func: A mapping function for data x.
            parameter_prefix: The prefix used if default parameters are generated.
            insert_barriers: If True, barriers are inserted in between the evolution instructions
                and hadamard layers.

        Raises:
            ValueError: If the feature dimension is smaller than 2.
        