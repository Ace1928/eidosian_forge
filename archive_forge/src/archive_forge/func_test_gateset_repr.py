import cirq
import cirq_ionq as ionq
import pytest
import sympy
@pytest.mark.parametrize('g', [ionq.IonQTargetGateset(), ionq.IonQTargetGateset(atol=1e-05)])
def test_gateset_repr(g):
    cirq.testing.assert_equivalent_repr(g, setup_code='import cirq_ionq\n')