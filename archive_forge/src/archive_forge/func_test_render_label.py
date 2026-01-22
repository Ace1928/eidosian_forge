from sympy.physics.quantum.circuitplot import labeller, render_label, Mz, CreateOneQubitGate,\
from sympy.physics.quantum.gate import CNOT, H, SWAP, CGate, S, T
from sympy.external import import_module
from sympy.testing.pytest import skip
def test_render_label():
    assert render_label('q0') == '$\\left|q0\\right\\rangle$'
    assert render_label('q0', {'q0': '0'}) == '$\\left|q0\\right\\rangle=\\left|0\\right\\rangle$'