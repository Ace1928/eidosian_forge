import cirq
import pytest
@pytest.mark.parametrize('cv1, cv2, expected_type', [[cirq.ProductOfSums([0, 1]), cirq.ProductOfSums([0]), cirq.ProductOfSums], [cirq.SumOfProducts([(0, 0), (1, 1)]), cirq.ProductOfSums([0, 1, 2]), cirq.SumOfProducts]])
def test_and_operation_adds_qubits(cv1, cv2, expected_type):
    assert isinstance(cv1 & cv2, expected_type)
    assert cirq.num_qubits(cv1 & cv2) == cirq.num_qubits(cv1) + cirq.num_qubits(cv2)