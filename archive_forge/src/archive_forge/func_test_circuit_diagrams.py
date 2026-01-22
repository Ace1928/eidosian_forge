import itertools
import pytest
import cirq
import cirq.contrib.acquaintance as cca
@pytest.mark.parametrize('subgraph,part_size', itertools.product(cca.BipartiteGraphType, range(1, 5)))
def test_circuit_diagrams(part_size, subgraph):
    qubits = cirq.LineQubit.range(2 * part_size)
    gate = cca.BipartiteSwapNetworkGate(subgraph, part_size)
    circuit = cirq.Circuit(gate(*qubits))
    diagram = circuit_diagrams['undecomposed', subgraph, part_size]
    cirq.testing.assert_has_diagram(circuit, diagram)
    no_decomp = lambda op: isinstance(op.gate, (cca.AcquaintanceOpportunityGate, cca.SwapPermutationGate))
    circuit = cirq.expand_composite(circuit, no_decomp=no_decomp)
    diagram = circuit_diagrams['decomposed', subgraph, part_size]
    cirq.testing.assert_has_diagram(circuit, diagram)