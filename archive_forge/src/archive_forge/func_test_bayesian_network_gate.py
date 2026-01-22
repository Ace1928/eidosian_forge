import cirq
from cirq.testing import assert_json_roundtrip_works
from cirq.contrib.json import DEFAULT_CONTRIB_RESOLVERS
from cirq.contrib.acquaintance import SwapPermutationGate
from cirq.contrib.bayesian_network import BayesianNetworkGate
from cirq.contrib.quantum_volume import QuantumVolumeResult
def test_bayesian_network_gate():
    gate = BayesianNetworkGate(init_probs=[('q0', 0.125), ('q1', None)], arc_probs=[('q1', ('q0',), [0.25, 0.5])])
    assert_json_roundtrip_works(gate, resolvers=DEFAULT_CONTRIB_RESOLVERS)