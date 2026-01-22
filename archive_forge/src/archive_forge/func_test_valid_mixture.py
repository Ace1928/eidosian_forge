import pytest
import numpy as np
import cirq
def test_valid_mixture():
    cirq.validate_mixture(ReturnsValidTuple())