import pytest
from cirq import quirk_url_to_circuit
def test_non_physical_operations():
    with pytest.raises(NotImplementedError, match='unphysical operation'):
        _ = quirk_url_to_circuit('https://algassert.com/quirk#circuit={"cols":[["__error__"]]}')
    with pytest.raises(NotImplementedError, match='unphysical operation'):
        _ = quirk_url_to_circuit('https://algassert.com/quirk#circuit={"cols":[["__unstable__UniversalNot"]]}')