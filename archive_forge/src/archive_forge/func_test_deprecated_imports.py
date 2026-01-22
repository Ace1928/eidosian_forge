from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import symbols
from sympy.core.singleton import S
from sympy.core.function import expand, Function
from sympy.core.numbers import I
from sympy.integrals.integrals import Integral
from sympy.polys.polytools import factor
from sympy.core.traversal import preorder_traversal, use, postorder_traversal, iterargs, iterfreeargs
from sympy.functions.elementary.piecewise import ExprCondPair, Piecewise
from sympy.testing.pytest import warns_deprecated_sympy
from sympy.utilities.iterables import capture
def test_deprecated_imports():
    x = symbols('x')
    with warns_deprecated_sympy():
        from sympy.core.basic import preorder_traversal
        preorder_traversal(x)
    with warns_deprecated_sympy():
        from sympy.simplify.simplify import bottom_up
        bottom_up(x, lambda x: x)
    with warns_deprecated_sympy():
        from sympy.simplify.simplify import walk
        walk(x, lambda x: x)
    with warns_deprecated_sympy():
        from sympy.simplify.traversaltools import use
        use(x, lambda x: x)
    with warns_deprecated_sympy():
        from sympy.utilities.iterables import postorder_traversal
        postorder_traversal(x)
    with warns_deprecated_sympy():
        from sympy.utilities.iterables import interactive_traversal
        capture(lambda: interactive_traversal(x))