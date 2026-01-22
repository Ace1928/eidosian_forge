from sympy.holonomic.recurrence import RecurrenceOperators, RecurrenceOperator
from sympy.core.symbol import symbols
from sympy.polys.domains.rationalfield import QQ
def test_RecurrenceOperatorEqPoly():
    n = symbols('n', integer=True)
    R, Sn = RecurrenceOperators(QQ.old_poly_ring(n), 'Sn')
    rr = RecurrenceOperator([n ** 2, 0, 0], R)
    rr2 = RecurrenceOperator([n ** 2, 1, n], R)
    assert not rr == rr2
    d2 = rr2.listofpoly[0]
    assert not rr2 == d2