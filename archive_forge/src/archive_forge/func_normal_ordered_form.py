import warnings
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.power import Pow
from sympy.physics.quantum import Operator, Commutator, AntiCommutator
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.fermion import FermionOp
def normal_ordered_form(expr, independent=False, recursive_limit=10, _recursive_depth=0):
    """Write an expression with bosonic or fermionic operators on normal
    ordered form, where each term is normally ordered. Note that this
    normal ordered form is equivalent to the original expression.

    Parameters
    ==========

    expr : expression
        The expression write on normal ordered form.

    recursive_limit : int (default 10)
        The number of allowed recursive applications of the function.

    Examples
    ========

    >>> from sympy.physics.quantum import Dagger
    >>> from sympy.physics.quantum.boson import BosonOp
    >>> from sympy.physics.quantum.operatorordering import normal_ordered_form
    >>> a = BosonOp("a")
    >>> normal_ordered_form(a * Dagger(a))
    1 + Dagger(a)*a
    """
    if _recursive_depth > recursive_limit:
        warnings.warn('Too many recursions, aborting')
        return expr
    if isinstance(expr, Add):
        return _normal_ordered_form_terms(expr, recursive_limit=recursive_limit, _recursive_depth=_recursive_depth, independent=independent)
    elif isinstance(expr, Mul):
        return _normal_ordered_form_factor(expr, recursive_limit=recursive_limit, _recursive_depth=_recursive_depth, independent=independent)
    else:
        return expr