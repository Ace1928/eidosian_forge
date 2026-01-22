import pytest
import numpy as np
import cirq
def test_assert_consistent_mixture_valid():
    mixture = cirq.X.with_probability(0.1)
    cirq.testing.assert_consistent_mixture(mixture)