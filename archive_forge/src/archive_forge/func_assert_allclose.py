import numpy.testing
import cupy
def assert_allclose(actual, desired, rtol=1e-07, atol=0, err_msg='', verbose=True):
    """Raises an AssertionError if objects are not equal up to desired tolerance.

    Args:
         actual(numpy.ndarray or cupy.ndarray): The actual object to check.
         desired(numpy.ndarray or cupy.ndarray): The desired, expected object.
         rtol(float): Relative tolerance.
         atol(float): Absolute tolerance.
         err_msg(str): The error message to be printed in case of failure.
         verbose(bool): If ``True``, the conflicting
             values are appended to the error message.

    .. seealso:: :func:`numpy.testing.assert_allclose`

    """
    numpy.testing.assert_allclose(cupy.asnumpy(actual), cupy.asnumpy(desired), rtol=rtol, atol=atol, err_msg=err_msg, verbose=verbose)