import numpy
import cupy
def isfortran(a):
    """Returns True if the array is Fortran contiguous but *not* C contiguous.

    If you only want to check if an array is Fortran contiguous use
    ``a.flags.f_contiguous`` instead.

    Args:
        a (cupy.ndarray): Input array.

    Returns:
        bool: The return value, True if ``a`` is Fortran contiguous but not C
        contiguous.

    .. seealso::
       :func:`~numpy.isfortran`

    Examples
    --------

    cupy.array allows to specify whether the array is written in C-contiguous
    order (last index varies the fastest), or FORTRAN-contiguous order in
    memory (first index varies the fastest).

    >>> a = cupy.array([[1, 2, 3], [4, 5, 6]], order='C')
    >>> a
    array([[1, 2, 3],
           [4, 5, 6]])
    >>> cupy.isfortran(a)
    False

    >>> b = cupy.array([[1, 2, 3], [4, 5, 6]], order='F')
    >>> b
    array([[1, 2, 3],
           [4, 5, 6]])
    >>> cupy.isfortran(b)
    True

    The transpose of a C-ordered array is a FORTRAN-ordered array.

    >>> a = cupy.array([[1, 2, 3], [4, 5, 6]], order='C')
    >>> a
    array([[1, 2, 3],
           [4, 5, 6]])
    >>> cupy.isfortran(a)
    False
    >>> b = a.T
    >>> b
    array([[1, 4],
           [2, 5],
           [3, 6]])
    >>> cupy.isfortran(b)
    True

    C-ordered arrays evaluate as False even if they are also FORTRAN-ordered.

    >>> cupy.isfortran(np.array([1, 2], order='F'))
    False

    """
    return a.flags.f_contiguous and (not a.flags.c_contiguous)