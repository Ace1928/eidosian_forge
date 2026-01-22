from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.functions.combinatorial.factorials import factorial
def shanks(A, k, n, m=1):
    """
    Calculate an approximation for lim k->oo A(k) using the n-term Shanks
    transformation S(A)(n). With m > 1, calculate the m-fold recursive
    Shanks transformation S(S(...S(A)...))(n).

    The Shanks transformation is useful for summing Taylor series that
    converge slowly near a pole or singularity, e.g. for log(2):

        >>> from sympy.abc import k, n
        >>> from sympy import Sum, Integer
        >>> from sympy.series.acceleration import shanks
        >>> A = Sum(Integer(-1)**(k+1) / k, (k, 1, n))
        >>> print(round(A.subs(n, 100).doit().evalf(), 10))
        0.6881721793
        >>> print(round(shanks(A, n, 25).evalf(), 10))
        0.6931396564
        >>> print(round(shanks(A, n, 25, 5).evalf(), 10))
        0.6931471806

    The correct value is 0.6931471805599453094172321215.
    """
    table = [A.subs(k, Integer(j)).doit() for j in range(n + m + 2)]
    table2 = table[:]
    for i in range(1, m + 1):
        for j in range(i, n + m + 1):
            x, y, z = (table[j - 1], table[j], table[j + 1])
            table2[j] = (z * x - y ** 2) / (z + x - 2 * y)
        table = table2[:]
    return table[n]