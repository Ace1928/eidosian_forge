import numpy as np
import pytest
import cirq
import cirq.testing
def assert_dirac_notation_numpy(vec, expected, decimals=2):
    assert cirq.dirac_notation(np.array(vec), decimals=decimals) == expected