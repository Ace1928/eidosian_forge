from collections import namedtuple
from decimal import Decimal
import numpy as np
from . import backends, blas, helpers, parser, paths, sharing
def contract_expression(subscripts, *shapes, **kwargs):
    """Generate a reusable expression for a given contraction with
    specific shapes, which can, for example, be cached.

    Parameters
    ----------
    subscripts : str
        Specifies the subscripts for summation.
    shapes : sequence of integer tuples
        Shapes of the arrays to optimize the contraction for.
    constants : sequence of int, optional
        The indices of any constant arguments in ``shapes``, in which case the
        actual array should be supplied at that position rather than just a
        shape. If these are specified, then constant parts of the contraction
        between calls will be reused. Additionally, if a GPU-enabled backend is
        used for example, then the constant tensors will be kept on the GPU,
        minimizing transfers.
    kwargs :
        Passed on to ``contract_path`` or ``einsum``. See ``contract``.

    Returns
    -------
    expr : ContractExpression
        Callable with signature ``expr(*arrays, out=None, backend='numpy')``
        where the array's shapes should match ``shapes``.

    Notes
    -----
    - The `out` keyword argument should be supplied to the generated expression
      rather than this function.
    - The `backend` keyword argument should also be supplied to the generated
      expression. If numpy arrays are supplied, if possible they will be
      converted to and back from the correct backend array type.
    - The generated expression will work with any arrays which have
      the same rank (number of dimensions) as the original shapes, however, if
      the actual sizes are different, the expression may no longer be optimal.
    - Constant operations will be computed upon the first call with a particular
      backend, then subsequently reused.

    Examples
    --------

    Basic usage:

        >>> expr = contract_expression("ab,bc->ac", (3, 4), (4, 5))
        >>> a, b = np.random.rand(3, 4), np.random.rand(4, 5)
        >>> c = expr(a, b)
        >>> np.allclose(c, a @ b)
        True

    Supply ``a`` as a constant:

        >>> expr = contract_expression("ab,bc->ac", a, (4, 5), constants=[0])
        >>> expr
        <ContractExpression('[ab],bc->ac', constants=[0])>

        >>> c = expr(b)
        >>> np.allclose(c, a @ b)
        True

    """
    if not kwargs.get('optimize', True):
        raise ValueError('Can only generate expressions for optimized contractions.')
    for arg in ('out', 'backend'):
        if kwargs.get(arg, None) is not None:
            raise ValueError("'{}' should only be specified when calling a `ContractExpression`, not when building it.".format(arg))
    if not isinstance(subscripts, str):
        subscripts, shapes = parser.convert_interleaved_input((subscripts,) + shapes)
    kwargs['_gen_expression'] = True
    constants = kwargs.pop('constants', ())
    constants_dict = {i: shapes[i] for i in constants}
    kwargs['_constants_dict'] = constants_dict
    dummy_arrays = [s if i in constants else shape_only(s) for i, s in enumerate(shapes)]
    return contract(subscripts, *dummy_arrays, **kwargs)