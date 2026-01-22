from sympy.utilities import dict_merge
from sympy.utilities.iterables import iterable
from sympy.physics.vector import (Dyadic, Vector, ReferenceFrame,
from sympy.physics.vector.printing import (vprint, vsprint, vpprint, vlatex,
from sympy.physics.mechanics.particle import Particle
from sympy.physics.mechanics.rigidbody import RigidBody
from sympy.simplify.simplify import simplify
from sympy.core.backend import (Matrix, sympify, Mul, Derivative, sin, cos,
def find_dynamicsymbols(expression, exclude=None, reference_frame=None):
    """Find all dynamicsymbols in expression.

    Explanation
    ===========

    If the optional ``exclude`` kwarg is used, only dynamicsymbols
    not in the iterable ``exclude`` are returned.
    If we intend to apply this function on a vector, the optional
    ``reference_frame`` is also used to inform about the corresponding frame
    with respect to which the dynamic symbols of the given vector is to be
    determined.

    Parameters
    ==========

    expression : SymPy expression

    exclude : iterable of dynamicsymbols, optional

    reference_frame : ReferenceFrame, optional
        The frame with respect to which the dynamic symbols of the
        given vector is to be determined.

    Examples
    ========

    >>> from sympy.physics.mechanics import dynamicsymbols, find_dynamicsymbols
    >>> from sympy.physics.mechanics import ReferenceFrame
    >>> x, y = dynamicsymbols('x, y')
    >>> expr = x + x.diff()*y
    >>> find_dynamicsymbols(expr)
    {x(t), y(t), Derivative(x(t), t)}
    >>> find_dynamicsymbols(expr, exclude=[x, y])
    {Derivative(x(t), t)}
    >>> a, b, c = dynamicsymbols('a, b, c')
    >>> A = ReferenceFrame('A')
    >>> v = a * A.x + b * A.y + c * A.z
    >>> find_dynamicsymbols(v, reference_frame=A)
    {a(t), b(t), c(t)}

    """
    t_set = {dynamicsymbols._t}
    if exclude:
        if iterable(exclude):
            exclude_set = set(exclude)
        else:
            raise TypeError('exclude kwarg must be iterable')
    else:
        exclude_set = set()
    if isinstance(expression, Vector):
        if reference_frame is None:
            raise ValueError('You must provide reference_frame when passing a vector expression, got %s.' % reference_frame)
        else:
            expression = expression.to_matrix(reference_frame)
    return {i for i in expression.atoms(AppliedUndef, Derivative) if i.free_symbols == t_set} - exclude_set