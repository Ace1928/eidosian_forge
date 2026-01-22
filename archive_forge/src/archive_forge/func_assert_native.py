import os
import platform
def assert_native(n):
    """Check whether the input is of native :py:class:`str` type.

    Raises:
        TypeError: in case of failed check

    """
    if not isinstance(n, str):
        raise TypeError('n must be a native str (got %s)' % type(n).__name__)