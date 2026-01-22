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
def test_run_sweeps_param_resolvers(dtype: Type[np.complexfloating], split: bool):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.DensityMatrixSimulator(dtype=dtype, split_untangled_states=split)
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            circuit = cirq.Circuit((cirq.X ** sympy.Symbol('b0'))(q0), (cirq.X ** sympy.Symbol('b1'))(q1), cirq.measure(q0), cirq.measure(q1))
            params = [cirq.ParamResolver({'b0': b0, 'b1': b1}), cirq.ParamResolver({'b0': b1, 'b1': b0})]
            results = simulator.run_sweep(circuit, params=params)
            assert len(results) == 2
            np.testing.assert_equal(results[0].measurements, {'q(0)': [[b0]], 'q(1)': [[b1]]})
            np.testing.assert_equal(results[1].measurements, {'q(0)': [[b1]], 'q(1)': [[b0]]})
            assert results[0].params == params[0]
            assert results[1].params == params[1]