import math
import cirq
import pytest
import numpy as np
import cirq_web
def test_bloch_sphere_default_client_code_comp_basis():
    bloch_sphere = cirq_web.BlochSphere(state_vector=1)
    expected_client_code = f"\n        <script>\n        renderBlochSphere('{bloch_sphere.id}', 5)\n            .addVector(0.0, 0.0, -1.0);\n        </script>\n        "
    assert expected_client_code == bloch_sphere.get_client_code()