import pytest
import cirq
@pytest.mark.parametrize('p', [-0.1, 1.1])
def test_validate_probability_invalid(p):
    with pytest.raises(ValueError, match='p'):
        cirq.validate_probability(p, 'p')