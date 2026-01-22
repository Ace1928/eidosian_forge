import math
import cirq
import pytest
import numpy as np
import cirq_web
@pytest.mark.parametrize('sphere_radius', [0, -1])
def test_invalid_bloch_sphere_radius(sphere_radius):
    with pytest.raises(ValueError):
        cirq_web.BlochSphere(sphere_radius=sphere_radius)