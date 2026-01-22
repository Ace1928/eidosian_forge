import cirq
import pytest
@pytest.mark.parametrize('data, is_trivial', [[((1,),), True], [((0, 1),), False], [((0, 0), (0, 1), (1, 0)), False], [((1, 1, 1, 1),), True]])
def test_sum_of_products_is_trivial(data, is_trivial):
    assert cirq.SumOfProducts(data).is_trivial == is_trivial