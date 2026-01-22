from typing import cast, Any
import numpy as np
from cirq._doc import document
Interpret an object as a pseudorandom number generator.

    If `random_state` is None, returns the module `np.random`.
    If `random_state` is an integer, returns
    `np.random.RandomState(random_state)`.
    Otherwise, returns `random_state` unmodified.

    Args:
        random_state: The object to be used as or converted to a pseudorandom
            number generator.

    Returns:
        The pseudorandom number generator object.
    