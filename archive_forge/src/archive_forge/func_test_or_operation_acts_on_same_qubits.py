import cirq
import pytest
@pytest.mark.parametrize('cv1,cv2,expected_type', [[cirq.ProductOfSums([0, 1, 2]), cirq.ProductOfSums([2, 0, 1]), cirq.ProductOfSums], [cirq.SumOfProducts([[0, 0, 0], [1, 1, 1]]), cirq.SumOfProducts([[0, 1, 0], [1, 0, 1]]), cirq.SumOfProducts], [cirq.ProductOfSums([0, 1]), cirq.ProductOfSums([1]), None], [cirq.SumOfProducts([[0], [1], [2]]), cirq.SumOfProducts([[1, 0]]), None]])
def test_or_operation_acts_on_same_qubits(cv1, cv2, expected_type):
    if cirq.num_qubits(cv1) == cirq.num_qubits(cv2):
        assert cirq.num_qubits(cv1 | cv2) == cirq.num_qubits(cv1)
        assert expected_type is not None
        assert isinstance(cv1 | cv2, expected_type)
    else:
        assert expected_type is None
        with pytest.raises(ValueError, match='must act on equal number of qubits'):
            _ = cv1 | cv2