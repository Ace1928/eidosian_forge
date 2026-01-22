import cirq
import pytest
def test_or_operation():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(cirq.ProductOfSums([0]) | cirq.ProductOfSums([1]), cirq.ProductOfSums([1]) | cirq.SumOfProducts([[0]]), cirq.SumOfProducts([[0]]) | cirq.ProductOfSums([1]), cirq.ProductOfSums([[0, 1]]), cirq.SumOfProducts([[1], [0]]))
    eq.add_equality_group(cirq.ProductOfSums([[0, 1]]) | cirq.ProductOfSums([2]), cirq.SumOfProducts([[0], [2]]) | cirq.ProductOfSums([1]), cirq.ProductOfSums([[0, 1, 2]]), cirq.SumOfProducts([[0], [1], [2]]))
    eq.add_equality_group(cirq.ProductOfSums([0, 1]) | cirq.SumOfProducts([[0, 0], [1, 1]]), cirq.SumOfProducts([[0, 0], [1, 1]]) | cirq.ProductOfSums([0, 1]), cirq.SumOfProducts([[0, 0], [1, 1], [0, 1]]))