from sympy.physics.quantum.qasm import Qasm, flip_index, trim,\
from sympy.physics.quantum.gate import X, Z, H, S, T
from sympy.physics.quantum.gate import CNOT, SWAP, CPHASE, CGate, CGateS
from sympy.physics.quantum.circuitplot import Mz
def test_qasm_3q():
    q = Qasm('qubit q0', 'qubit q1', 'qubit q2', 'toffoli q2,q1,q0')
    assert q.get_circuit() == CGateS((0, 1), X(2))