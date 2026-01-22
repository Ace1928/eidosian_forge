import cirq
import pytest
@pytest.mark.parametrize('data', [((1,),), ((0, 1),), ((0, 0), (0, 1), (1, 0))])
def test_sum_of_products_num_qubits(data):
    assert cirq.num_qubits(cirq.SumOfProducts(data)) == len(data[0])