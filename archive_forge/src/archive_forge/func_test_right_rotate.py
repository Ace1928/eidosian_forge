import cirq
from cirq.interop.quirk.cells.qubit_permutation_cells import QuirkQubitPermutationGate
from cirq.interop.quirk.cells.testing import assert_url_to_circuit_returns
def test_right_rotate():
    assert_url_to_circuit_returns('{"cols":[["X",">>4",1,1,1,"X"]]}', diagram='\n0: ───X───────────────────\n\n1: ───right_rotate[0>3]───\n      │\n2: ───right_rotate[1>0]───\n      │\n3: ───right_rotate[2>1]───\n      │\n4: ───right_rotate[3>2]───\n\n5: ───X───────────────────\n        ', maps={0: 33, 2: 37, 4: 41, 8: 49, 16: 35, 30: 63, 20: 43})