import numpy as np
import pytest
import sympy
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
@pytest.mark.parametrize('theta,exp', [(sympy.Symbol('theta'), 1 / 2), (np.pi / 2, 1 / 2), (np.pi / 2, sympy.Symbol('exp')), (sympy.Symbol('theta'), sympy.Symbol('exp'))])
def test_rxyz_exponent(theta, exp):

    def resolve(gate):
        return cirq.resolve_parameters(gate, {'theta': np.pi / 4}, {'exp': 1 / 4})
    assert resolve(cirq.Rx(rads=theta) ** exp) == resolve(cirq.Rx(rads=theta * exp))
    assert resolve(cirq.Ry(rads=theta) ** exp) == resolve(cirq.Ry(rads=theta * exp))
    assert resolve(cirq.Rz(rads=theta) ** exp) == resolve(cirq.Rz(rads=theta * exp))