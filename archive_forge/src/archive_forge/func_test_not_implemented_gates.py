import pytest
from cirq import quirk_url_to_circuit
def test_not_implemented_gates():
    for k in ['X^⌈t⌉', 'X^⌈t-¼⌉', 'Counting4', 'Uncounting4', '>>t3', '<<t3']:
        with pytest.raises(NotImplementedError, match='discrete parameter'):
            _ = quirk_url_to_circuit('https://algassert.com/quirk#circuit={"cols":[["' + k + '"]]}')
    for k in ['add3', 'sub3', 'c+=ab4', 'c-=ab4']:
        with pytest.raises(NotImplementedError, match='deprecated'):
            _ = quirk_url_to_circuit('https://algassert.com/quirk#circuit={"cols":[["' + k + '"]]}')
    for k in ['X', 'Y', 'Z']:
        with pytest.raises(NotImplementedError, match='feedback'):
            _ = quirk_url_to_circuit('https://algassert.com/quirk#circuit={"cols":[["' + k + 'DetectControlReset"]]}')
    for k in ['|0⟩⟨0|', '|1⟩⟨1|', '|+⟩⟨+|', '|-⟩⟨-|', '|X⟩⟨X|', '|/⟩⟨/|', '0']:
        with pytest.raises(NotImplementedError, match='postselection'):
            _ = quirk_url_to_circuit('https://algassert.com/quirk#circuit={"cols":[["' + k + '"]]}')