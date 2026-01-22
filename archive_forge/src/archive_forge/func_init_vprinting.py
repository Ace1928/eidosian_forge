from sympy.core.function import Derivative
from sympy.core.function import UndefinedFunction, AppliedUndef
from sympy.core.symbol import Symbol
from sympy.interactive.printing import init_printing
from sympy.printing.latex import LatexPrinter
from sympy.printing.pretty.pretty import PrettyPrinter
from sympy.printing.pretty.pretty_symbology import center_accent
from sympy.printing.str import StrPrinter
from sympy.printing.precedence import PRECEDENCE
def init_vprinting(**kwargs):
    """Initializes time derivative printing for all SymPy objects, i.e. any
    functions of time will be displayed in a more compact notation. The main
    benefit of this is for printing of time derivatives; instead of
    displaying as ``Derivative(f(t),t)``, it will display ``f'``. This is
    only actually needed for when derivatives are present and are not in a
    physics.vector.Vector or physics.vector.Dyadic object. This function is a
    light wrapper to :func:`~.init_printing`. Any keyword
    arguments for it are valid here.

    {0}

    Examples
    ========

    >>> from sympy import Function, symbols
    >>> t, x = symbols('t, x')
    >>> omega = Function('omega')
    >>> omega(x).diff()
    Derivative(omega(x), x)
    >>> omega(t).diff()
    Derivative(omega(t), t)

    Now use the string printer:

    >>> from sympy.physics.vector import init_vprinting
    >>> init_vprinting(pretty_print=False)
    >>> omega(x).diff()
    Derivative(omega(x), x)
    >>> omega(t).diff()
    omega'

    """
    kwargs['str_printer'] = vsstrrepr
    kwargs['pretty_printer'] = vpprint
    kwargs['latex_printer'] = vlatex
    init_printing(**kwargs)