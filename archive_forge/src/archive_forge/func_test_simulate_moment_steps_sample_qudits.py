import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
@pytest.mark.parametrize('split', [True, False])
def test_simulate_moment_steps_sample_qudits(dtype: Type[np.complexfloating], split: bool):

    class TestGate(cirq.Gate):
        """Swaps the 2nd qid |0> and |2> states when the 1st is |1>."""

        def _qid_shape_(self):
            return (2, 3)

        def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs):
            args.available_buffer[..., 1, 0] = args.target_tensor[..., 1, 2]
            args.target_tensor[..., 1, 2] = args.target_tensor[..., 1, 0]
            args.target_tensor[..., 1, 0] = args.available_buffer[..., 1, 0]
            return args.target_tensor
    q0, q1 = cirq.LineQid.for_qid_shape((2, 3))
    circuit = cirq.Circuit(cirq.H(q0), TestGate()(q0, q1))
    simulator = cirq.DensityMatrixSimulator(dtype=dtype, split_untangled_states=split)
    for i, step in enumerate(simulator.simulate_moment_steps(circuit)):
        if i == 0:
            samples = step.sample([q0, q1], repetitions=10)
            for sample in samples:
                assert np.array_equal(sample, [True, 0]) or np.array_equal(sample, [False, 0])
        else:
            samples = step.sample([q0, q1], repetitions=10)
            for sample in samples:
                assert np.array_equal(sample, [True, 2]) or np.array_equal(sample, [False, 0])