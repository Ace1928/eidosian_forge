from sympy.core.numbers import Integer
from sympy.core.symbol import Symbol
from sympy.physics.quantum.qexpr import QExpr, _qsympify_sequence
from sympy.physics.quantum.hilbert import HilbertSpace
from sympy.core.containers import Tuple
def test_qexpr_commutative():
    q1 = QExpr(x)
    q2 = QExpr(y)
    assert q1.is_commutative is False
    assert q2.is_commutative is False
    assert q1 * q2 != q2 * q1
    q = QExpr._new_rawargs(Integer(0), Integer(1), HilbertSpace())
    assert q.is_commutative is False