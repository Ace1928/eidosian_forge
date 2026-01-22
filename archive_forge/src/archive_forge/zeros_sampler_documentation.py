import abc
from typing import List, Optional, TYPE_CHECKING
import numpy as np
from cirq import devices, work, study
Samples circuit as if every measurement resulted in zero.

        Args:
            program: The circuit to sample from.
            params: Parameters to run with the program.
            repetitions: The number of times to sample.

        Returns:
            Result list for this run; one for each possible parameter
            resolver.

        Raises:
            ValueError: circuit is not valid for the sampler, due to invalid
            repeated keys or incompatibility with the sampler's device.
        