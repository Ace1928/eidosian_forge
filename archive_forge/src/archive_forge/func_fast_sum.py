import typing
from typing import Iterable, Mapping, Union
from ortools.math_opt.python import model
def fast_sum(summands):
    """Sums the elements of summand into a linear or quadratic expression.

    Similar to Python's sum function, but faster for input that not just integers
    and floats.

    Unlike sum(), the function returns a linear expression when all inputs are
    floats and/or integers. Importantly, the code:
      model.add_linear_constraint(fast_sum(maybe_empty_list) <= 1.0)
    is safe to call, while:
      model.add_linear_constraint(sum(maybe_empty_list) <= 1.0)
    fails at runtime when the list is empty.

    Args:
      summands: The elements to add up.

    Returns:
      A linear or quadratic expression with the sum of the elements of summand.
    """
    summands_tuple = tuple(summands)
    for s in summands_tuple:
        if isinstance(s, model.QuadraticBase):
            return model.QuadraticSum(summands_tuple)
    return model.LinearSum(summands_tuple)