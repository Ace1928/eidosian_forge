import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
def test_pure_state_creation():
    sim = cirq.Simulator()
    qids = cirq.LineQubit.range(3)
    shape = cirq.qid_shape(qids)
    args = sim._create_simulation_state(1, qids)
    values = list(args.values())
    arg = values[0].kronecker_product(values[1]).kronecker_product(values[2]).transpose_to_qubit_order(qids)
    expected = cirq.to_valid_state_vector(1, len(qids), qid_shape=shape)
    np.testing.assert_allclose(arg.target_tensor, expected.reshape(shape))