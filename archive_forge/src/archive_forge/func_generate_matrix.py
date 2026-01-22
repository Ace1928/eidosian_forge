import contextlib
import inspect
from typing import Callable
import unittest
from unittest import mock
import warnings
import numpy
import cupy
from cupy._core import internal
import cupyx
import cupyx.scipy.sparse
from cupy.testing._pytest_impl import is_available
def generate_matrix(shape, xp=cupy, dtype=numpy.float32, *, singular_values=None):
    """Returns a matrix with specified singular values.

    Generates a random matrix with given singular values.
    This function generates a random NumPy matrix (or a stack of matrices) that
    has specified singular values. It can be used to generate the inputs for a
    test that can be instable when the input value behaves bad.
    Notation: denote the shape of the generated array by :math:`(B..., M, N)`,
    and :math:`K = min\\{M, N\\}`. :math:`B...` may be an empty sequence.

    Args:
        shape (tuple of int): Shape of the generated array, i.e.,
            :math:`(B..., M, N)`.
        xp (numpy or cupy): Array module to use.
        dtype: Dtype of the generated array.
        singular_values (array-like): Singular values of the generated
            matrices. It must be broadcastable to shape :math:`(B..., K)`.

    Returns:
        numpy.ndarray or cupy.ndarray: A random matrix that has specifiec
        singular values.
    """
    if len(shape) <= 1:
        raise ValueError('shape {} is invalid for matrices: too few axes'.format(shape))
    if singular_values is None:
        raise TypeError('singular_values is not given')
    singular_values = xp.asarray(singular_values)
    dtype = numpy.dtype(dtype)
    if dtype.kind not in 'fc':
        raise TypeError('dtype {} is not supported'.format(dtype))
    if not xp.isrealobj(singular_values):
        raise TypeError('singular_values is not real')
    if (singular_values < 0).any():
        raise ValueError('negative singular value is given')
    a = xp.random.randn(*shape)
    if dtype.kind == 'c':
        a = a + 1j * xp.random.randn(*shape)
    u, s, vh = xp.linalg.svd(a, full_matrices=False)
    sv = xp.broadcast_to(singular_values, s.shape)
    a = xp.einsum('...ik,...k,...kj->...ij', u, sv, vh)
    return a.astype(dtype)