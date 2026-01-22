import logging
import sys
from pyomo.common.dependencies import numpy_available
from pyomo.common.deprecation import deprecated, relocated_module_attribute
from pyomo.common.errors import TemplateExpressionError

    A utility function that returns the value of a Pyomo object or
    expression.

    Args:
        obj: The argument to evaluate. If it is None, a
            string, or any other primitive numeric type,
            then this function simply returns the argument.
            Otherwise, if the argument is a NumericValue
            then the __call__ method is executed.
        exception (bool): If :const:`True`, then an exception should
            be raised when instances of NumericValue fail to
            evaluate due to one or more objects not being
            initialized to a numeric value (e.g, one or more
            variables in an algebraic expression having the
            value None). If :const:`False`, then the function
            returns :const:`None` when an exception occurs.
            Default is True.

    Returns: A numeric value or None.
    