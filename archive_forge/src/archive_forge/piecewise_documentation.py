import cupy
from cupy import _core
Evaluate a piecewise-defined function.

        Args:
            x (cupy.ndarray): input domain
            condlist (list of cupy.ndarray):
                Each boolean array/ scalar corresponds to a function
                in funclist. Length of funclist is equal to that of
                condlist. If one extra function is given, it is used
                as the default value when the otherwise condition is met
            funclist (list of scalars): list of scalar functions.

        Returns:
            cupy.ndarray: the scalar values in funclist on portions of x
            defined by condlist.

        .. warning::

            This function currently doesn't support callable functions,
            args and kw parameters.

        .. seealso:: :func:`numpy.piecewise`
        