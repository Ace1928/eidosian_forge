from typing import Any, Callable, List, Optional, Tuple, Union, TYPE_CHECKING
import numbers
import sympy
from cirq import protocols
from cirq.study import resolver, sweeps, sweepable
def flatten_with_params(val: Any, params: resolver.ParamResolverOrSimilarType) -> Tuple[Any, resolver.ParamDictType]:
    """Creates a copy of `val` with any symbols or expressions replaced with
    new symbols.  `val` can be a `Circuit`, `Gate`, `Operation`, or other
    type.  Also transforms a dictionary of symbol values for `val` to an
    equivalent dictionary mapping the new symbols to their evaluated values.

    `flatten_with_params` goes through every parameter in `val` and does the
    following:
    - If the parameter is a number, don't change it.
    - If the parameter is a symbol, don't change it and use the same symbol with
        the same value in the new dictionary of symbol values.
    - If the parameter is an expression, replace it with a symbol and use the
        new symbol with the evaluated value of the expression in the new
        dictionary of symbol values.  The new symbol will be
        `sympy.Symbol('<x + 1>')` if the expression was `sympy.Symbol('x') + 1`.
        In the unlikely case that an expression with a different meaning also
        has the string `'x + 1'`, a number is appended to the name to avoid
        collision: `sympy.Symbol('<x + 1>_1')`.

    Args:
        val: The value to copy and substitute parameter expressions with
        flattened symbols.
        params: A dictionary or `ParamResolver` where the keys are
            `sympy.Symbol`s used by `val` and the values are numbers.

    Returns:
        The tuple (new value, new params) where new value is `val` with
        flattened expressions and new params is a dictionary mapping the
        new symbols like `sympy.Symbol('<x + 1>')` to numbers like
        `params['x'] + 1`.
    """
    val_flat, expr_map = flatten(val)
    new_params = expr_map.transform_params(params)
    return (val_flat, new_params)