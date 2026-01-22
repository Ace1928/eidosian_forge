import math
import cirq
import pytest
import numpy as np
import cirq_web
def test_init_bloch_sphere_type():
    bloch_sphere = cirq_web.BlochSphere(state_vector=[1 / math.sqrt(2), 1 / math.sqrt(2)])
    assert isinstance(bloch_sphere, cirq_web.BlochSphere)