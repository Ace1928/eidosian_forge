from symengine import Symbol, var
from symengine.test_utilities import raises
def test_var_global_namespace():
    raises(NameError, lambda: z1)
    _make_z1()
    assert z1 == Symbol('z1')
    raises(NameError, lambda: z2)
    _make_z2()
    assert z2 == Symbol('z2')