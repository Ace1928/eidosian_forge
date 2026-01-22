from .. import Backend
import pytest
@pytest.mark.parametrize('key', backends)
def test_basic_cse(key):
    be = Backend(key)
    x, y = map(be.Symbol, 'xy')
    exprs = [x ** 2 + y ** 2 + 3, be.exp(x ** 2 + y ** 2)]
    subs_cses, cse_exprs = be.cse(exprs)
    subs, cses = zip(*subs_cses)
    assert cses[0] == x ** 2 + y ** 2
    for cse_expr in cse_exprs:
        assert x not in cse_expr.atoms()
        assert y not in cse_expr.atoms()
    assert _inverse_cse(subs_cses, cse_exprs) == exprs