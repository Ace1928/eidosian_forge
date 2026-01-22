import numpy.testing
import cupy
def assert_array_less(x, y, err_msg='', verbose=True):
    """Raises an AssertionError if array_like objects are not ordered by less than.

    Args:
         x(numpy.ndarray or cupy.ndarray): The smaller object to check.
         y(numpy.ndarray or cupy.ndarray): The larger object to compare.
         err_msg(str): The error message to be printed in case of failure.
         verbose(bool): If ``True``, the conflicting values
             are appended to the error message.

    .. seealso:: :func:`numpy.testing.assert_array_less`
    """
    numpy.testing.assert_array_less(cupy.asnumpy(x), cupy.asnumpy(y), err_msg=err_msg, verbose=verbose)