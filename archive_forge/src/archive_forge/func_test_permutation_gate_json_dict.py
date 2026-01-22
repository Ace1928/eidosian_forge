import pytest
import cirq
import numpy as np
from cirq.ops import QubitPermutationGate
def test_permutation_gate_json_dict():
    assert cirq.QubitPermutationGate([0, 1, 2])._json_dict_() == {'permutation': (0, 1, 2)}