import numpy as np
import pytest
import cirq
import cirq.testing
def test_dirac_notation():
    sqrt = np.sqrt(0.5)
    exp_pi_2 = 0.5 + 0.5j
    assert_dirac_notation_numpy([0, 0], '0')
    assert_dirac_notation_python([1], '|⟩')
    assert_dirac_notation_numpy([sqrt, sqrt], '0.71|0⟩ + 0.71|1⟩')
    assert_dirac_notation_python([-sqrt, sqrt], '-0.71|0⟩ + 0.71|1⟩')
    assert_dirac_notation_numpy([sqrt, -sqrt], '0.71|0⟩ - 0.71|1⟩')
    assert_dirac_notation_python([-sqrt, -sqrt], '-0.71|0⟩ - 0.71|1⟩')
    assert_dirac_notation_numpy([sqrt, 1j * sqrt], '0.71|0⟩ + 0.71j|1⟩')
    assert_dirac_notation_python([sqrt, exp_pi_2], '0.71|0⟩ + (0.5+0.5j)|1⟩')
    assert_dirac_notation_numpy([exp_pi_2, -sqrt], '(0.5+0.5j)|0⟩ - 0.71|1⟩')
    assert_dirac_notation_python([exp_pi_2, 0.5 - 0.5j], '(0.5+0.5j)|0⟩ + (0.5-0.5j)|1⟩')
    assert_dirac_notation_numpy([0.5, 0.5, -0.5, -0.5], '0.5|00⟩ + 0.5|01⟩ - 0.5|10⟩ - 0.5|11⟩')
    assert_dirac_notation_python([0.71j, 0.71j], '0.71j|0⟩ + 0.71j|1⟩')