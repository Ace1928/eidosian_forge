import numpy.testing
import cupy
def assert_array_almost_equal(x, y, decimal=6, err_msg='', verbose=True):
    """Raises an AssertionError if objects are not equal up to desired precision.

    Args:
         x(numpy.ndarray or cupy.ndarray): The actual object to check.
         y(numpy.ndarray or cupy.ndarray): The desired, expected object.
         decimal(int): Desired precision.
         err_msg(str): The error message to be printed in case of failure.
         verbose(bool): If ``True``, the conflicting
             values are appended to the error message.

    .. seealso:: :func:`numpy.testing.assert_array_almost_equal`
    """
    numpy.testing.assert_array_almost_equal(cupy.asnumpy(x), cupy.asnumpy(y), decimal=decimal, err_msg=err_msg, verbose=verbose)