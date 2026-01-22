from sympy.physics.quantum.qasm import Qasm, flip_index, trim,\
from sympy.physics.quantum.gate import X, Z, H, S, T
from sympy.physics.quantum.gate import CNOT, SWAP, CPHASE, CGate, CGateS
from sympy.physics.quantum.circuitplot import Mz
def test_qasm_get_index():
    assert get_index('q0', ['q0', 'q1']) == 1
    assert get_index('q1', ['q0', 'q1']) == 0