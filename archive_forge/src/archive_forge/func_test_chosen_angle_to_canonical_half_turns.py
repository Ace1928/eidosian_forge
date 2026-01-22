import numpy as np
import pytest
import sympy
import cirq
def test_chosen_angle_to_canonical_half_turns():
    assert cirq.chosen_angle_to_canonical_half_turns() == 1
    assert cirq.chosen_angle_to_canonical_half_turns(default=0.5) == 0.5
    assert cirq.chosen_angle_to_canonical_half_turns(half_turns=0.25, default=0.75) == 0.25
    np.testing.assert_allclose(cirq.chosen_angle_to_canonical_half_turns(rads=np.pi / 2), 0.5, atol=1e-08)
    np.testing.assert_allclose(cirq.chosen_angle_to_canonical_half_turns(rads=-np.pi / 4), -0.25, atol=1e-08)
    assert cirq.chosen_angle_to_canonical_half_turns(degs=90) == 0.5
    assert cirq.chosen_angle_to_canonical_half_turns(degs=1080) == 0
    assert cirq.chosen_angle_to_canonical_half_turns(degs=990) == -0.5
    with pytest.raises(ValueError):
        _ = cirq.chosen_angle_to_canonical_half_turns(half_turns=0, rads=0)
    with pytest.raises(ValueError):
        _ = cirq.chosen_angle_to_canonical_half_turns(half_turns=0, degs=0)
    with pytest.raises(ValueError):
        _ = cirq.chosen_angle_to_canonical_half_turns(degs=0, rads=0)
    with pytest.raises(ValueError):
        _ = cirq.chosen_angle_to_canonical_half_turns(half_turns=0, rads=0, degs=0)