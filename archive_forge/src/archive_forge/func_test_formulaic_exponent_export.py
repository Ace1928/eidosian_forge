import pytest
import sympy
import cirq
from cirq.contrib.quirk.export_to_quirk import circuit_to_quirk_url
def test_formulaic_exponent_export():
    a = cirq.LineQubit(0)
    t = sympy.Symbol('t')
    assert_links_to(cirq.Circuit(cirq.X(a) ** t, cirq.Y(a) ** (-t), cirq.Z(a) ** (t * 2 + 1)), '\n        http://algassert.com/quirk#circuit={"cols":[\n            ["X^t"],\n            ["Y^-t"],\n            [{"arg":"2*t+1","id":"Z^ft"}]\n        ]}\n    ', escape_url=False)