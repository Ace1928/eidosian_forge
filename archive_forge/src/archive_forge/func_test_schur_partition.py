from sympy.core import S, Rational
from sympy.combinatorics.schur_number import schur_partition, SchurNumber
from sympy.core.random import _randint
from sympy.testing.pytest import raises
from sympy.core.symbol import symbols
def test_schur_partition():
    raises(ValueError, lambda: schur_partition(S.Infinity))
    raises(ValueError, lambda: schur_partition(-1))
    raises(ValueError, lambda: schur_partition(0))
    assert schur_partition(2) == [[1, 2]]
    random_number_generator = _randint(1000)
    for _ in range(5):
        n = random_number_generator(1, 1000)
        result = schur_partition(n)
        t = 0
        numbers = []
        for item in result:
            _sum_free_test(item)
            '\n            Checks if the occurrence of all numbers is exactly one\n            '
            t += len(item)
            for l in item:
                assert (l in numbers) is False
                numbers.append(l)
        assert n == t
    x = symbols('x')
    raises(ValueError, lambda: schur_partition(x))