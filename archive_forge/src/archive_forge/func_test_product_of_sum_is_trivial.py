import cirq
import pytest
@pytest.mark.parametrize('data, is_trivial', [[((1,),), True], [((0, 1),), False], [([2], [1], [2]), False], [([1], [1], [1], [1]), True]])
def test_product_of_sum_is_trivial(data, is_trivial):
    assert cirq.ProductOfSums(data).is_trivial == is_trivial