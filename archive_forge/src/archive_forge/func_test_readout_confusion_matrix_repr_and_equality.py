import numpy as np
import cirq
import pytest
from cirq.experiments.single_qubit_readout_calibration_test import NoisySingleQubitReadoutSampler
def test_readout_confusion_matrix_repr_and_equality():
    mat1 = cirq.testing.random_orthogonal(4, random_state=1234)
    mat2 = cirq.testing.random_orthogonal(2, random_state=1234)
    q = cirq.LineQubit.range(3)
    a = cirq.TensoredConfusionMatrices([mat1, mat2], [q[:2], q[2:]], repetitions=0, timestamp=0)
    b = cirq.TensoredConfusionMatrices(mat1, q[:2], repetitions=0, timestamp=0)
    c = cirq.TensoredConfusionMatrices(mat2, q[2:], repetitions=0, timestamp=0)
    for x in [a, b, c]:
        cirq.testing.assert_equivalent_repr(x)
        assert cirq.approx_eq(x, x)
        assert x._approx_eq_(mat1, 1e-06) is NotImplemented
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(a, a)
    eq.add_equality_group(b, b)
    eq.add_equality_group(c, c)