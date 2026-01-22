from random import random
from typing import Callable
import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from scipy.linalg import block_diag
import cirq
from cirq.transformers.analytical_decompositions.three_qubit_decomposition import (
Returns the CS matrix from the cosine sine decomposition.

    Args:
        theta: the 4 angles that result from the CS decomposition.
    Returns:
        the CS matrix
    