from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.functions import DiracDelta, Heaviside
from .integrals import Integral, integrate
def change_mul(node, x):
    """change_mul(node, x)

       Rearranges the operands of a product, bringing to front any simple
       DiracDelta expression.

       Explanation
       ===========

       If no simple DiracDelta expression was found, then all the DiracDelta
       expressions are simplified (using DiracDelta.expand(diracdelta=True, wrt=x)).

       Return: (dirac, new node)
       Where:
         o dirac is either a simple DiracDelta expression or None (if no simple
           expression was found);
         o new node is either a simplified DiracDelta expressions or None (if it
           could not be simplified).

       Examples
       ========

       >>> from sympy import DiracDelta, cos
       >>> from sympy.integrals.deltafunctions import change_mul
       >>> from sympy.abc import x, y
       >>> change_mul(x*y*DiracDelta(x)*cos(x), x)
       (DiracDelta(x), x*y*cos(x))
       >>> change_mul(x*y*DiracDelta(x**2 - 1)*cos(x), x)
       (None, x*y*cos(x)*DiracDelta(x - 1)/2 + x*y*cos(x)*DiracDelta(x + 1)/2)
       >>> change_mul(x*y*DiracDelta(cos(x))*cos(x), x)
       (None, None)

       See Also
       ========

       sympy.functions.special.delta_functions.DiracDelta
       deltaintegrate
    """
    new_args = []
    dirac = None
    c, nc = node.args_cnc()
    sorted_args = sorted(c, key=default_sort_key)
    sorted_args.extend(nc)
    for arg in sorted_args:
        if arg.is_Pow and isinstance(arg.base, DiracDelta):
            new_args.append(arg.func(arg.base, arg.exp - 1))
            arg = arg.base
        if dirac is None and (isinstance(arg, DiracDelta) and arg.is_simple(x)):
            dirac = arg
        else:
            new_args.append(arg)
    if not dirac:
        new_args = []
        for arg in sorted_args:
            if isinstance(arg, DiracDelta):
                new_args.append(arg.expand(diracdelta=True, wrt=x))
            elif arg.is_Pow and isinstance(arg.base, DiracDelta):
                new_args.append(arg.func(arg.base.expand(diracdelta=True, wrt=x), arg.exp))
            else:
                new_args.append(arg)
        if new_args != sorted_args:
            nnode = Mul(*new_args).expand()
        else:
            nnode = None
        return (None, nnode)
    return (dirac, Mul(*new_args))