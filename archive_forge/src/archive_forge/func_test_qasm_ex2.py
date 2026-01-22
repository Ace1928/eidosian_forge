from sympy.physics.quantum.qasm import Qasm, flip_index, trim,\
from sympy.physics.quantum.gate import X, Z, H, S, T
from sympy.physics.quantum.gate import CNOT, SWAP, CPHASE, CGate, CGateS
from sympy.physics.quantum.circuitplot import Mz
def test_qasm_ex2():
    q = Qasm('qubit q_0', 'qubit q_1', 'qubit q_2', 'h  q_1', 'cnot q_1,q_2', 'cnot q_0,q_1', 'h q_0', 'measure q_1', 'measure q_0', 'c-x q_1,q_2', 'c-z q_0,q_2')
    assert q.get_circuit() == CGate(2, Z(0)) * CGate(1, X(0)) * Mz(2) * Mz(1) * H(2) * CNOT(2, 1) * CNOT(1, 0) * H(1)