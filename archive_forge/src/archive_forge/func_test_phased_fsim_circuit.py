import numpy as np
import pytest
import sympy
import cirq
def test_phased_fsim_circuit():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.PhasedFSimGate(np.pi / 2, np.pi, np.pi / 2, 0, -np.pi / 4).on(a, b), cirq.PhasedFSimGate(-np.pi, np.pi / 2, np.pi / 10, np.pi / 5, 3 * np.pi / 10).on(a, b))
    cirq.testing.assert_has_diagram(c, '\n0: ───PhFSim(0.5π, -π, 0.5π, 0, -0.25π)───PhFSim(-π, 0.5π, 0.1π, 0.2π, 0.3π)───\n      │                                   │\n1: ───PhFSim(0.5π, -π, 0.5π, 0, -0.25π)───PhFSim(-π, 0.5π, 0.1π, 0.2π, 0.3π)───\n    ')
    cirq.testing.assert_has_diagram(c, '\n0: ---PhFSim(0.5pi, -pi, 0.5pi, 0, -0.25pi)---PhFSim(-pi, 0.5pi, 0.1pi, 0.2pi, 0.3pi)---\n      |                                       |\n1: ---PhFSim(0.5pi, -pi, 0.5pi, 0, -0.25pi)---PhFSim(-pi, 0.5pi, 0.1pi, 0.2pi, 0.3pi)---\n        ', use_unicode_characters=False)
    cirq.testing.assert_has_diagram(c, '\n0: ---PhFSim(1.5707963267948966, -pi, 1.5707963267948966, 0, -0.7853981633974483)---PhFSim(-pi, 1.5707963267948966, 0.3141592653589793, 0.6283185307179586, 0.9424777960769379)---\n      |                                                                             |\n1: ---PhFSim(1.5707963267948966, -pi, 1.5707963267948966, 0, -0.7853981633974483)---PhFSim(-pi, 1.5707963267948966, 0.3141592653589793, 0.6283185307179586, 0.9424777960769379)---\n', use_unicode_characters=False, precision=None)
    c = cirq.Circuit(cirq.PhasedFSimGate(sympy.Symbol('a') + sympy.Symbol('b'), 0, sympy.Symbol('c'), sympy.Symbol('d'), sympy.Symbol('a') - sympy.Symbol('b')).on(a, b))
    cirq.testing.assert_has_diagram(c, '\n0: ───PhFSim(a + b, 0, c, d, a - b)───\n      │\n1: ───PhFSim(a + b, 0, c, d, a - b)───\n    ')