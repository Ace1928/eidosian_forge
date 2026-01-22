import numpy as np
import pytest
import cirq
from cirq import value
from cirq.transformers.heuristic_decompositions.gate_tabulation_math_utils import (
def test_weyl_chamber_mesh_spacing_too_small_throws_error():
    with pytest.raises(ValueError, match='may cause system to crash'):
        weyl_chamber_mesh(spacing=0.0005)