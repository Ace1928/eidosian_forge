import cirq
import pytest
def test_init_sum_of_products():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(cirq.SumOfProducts([[1]]), cirq.SumOfProducts(((1,),)))
    eq.add_equality_group(cirq.SumOfProducts([[0]]), cirq.SumOfProducts(((0,),)))
    eq.add_equality_group(cirq.SumOfProducts([[0], [1], [2]], name='custom name'), cirq.SumOfProducts([[0], [1], [2]], name='name does not matter'), cirq.SumOfProducts([[2], [0], [1]]), cirq.SumOfProducts([[0], [0], [2], [2], [1], [1]]))
    eq.add_equality_group(cirq.SumOfProducts([[0, 1, 2]]), cirq.SumOfProducts([(0, 1, 2)]))
    eq.add_equality_group(cirq.SumOfProducts([[1, 0, 2]]))
    eq.add_equality_group(cirq.SumOfProducts([(0, 2, 1)]))
    eq.add_equality_group(cirq.SumOfProducts([[0, 0, 1, 1, 2, 2]]))
    eq.add_equality_group(cirq.SumOfProducts([(0, 1), (0, 2), (1, 1), (1, 2)]), cirq.SumOfProducts([(1, 2), (0, 2), (0, 1), (1, 1)]), cirq.SumOfProducts([(1, 2), (1, 2), (0, 2), (0, 2), (1, 1), (0, 1)]))
    eq.add_equality_group(cirq.SumOfProducts([(1, 0), (2, 0), (1, 1), (2, 1)]))