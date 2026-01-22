import pytest
from pytest import param
from numpy.testing import IS_WASM
import numpy as np
def check_operations(dtype, value):
    """
    There are many dedicated paths in NumPy which cast and should check for
    floating point errors which occurred during those casts.
    """
    if dtype.kind != 'i':

        def assignment():
            arr = np.empty(3, dtype=dtype)
            arr[0] = value
        yield assignment

        def fill():
            arr = np.empty(3, dtype=dtype)
            arr.fill(value)
        yield fill

    def copyto_scalar():
        arr = np.empty(3, dtype=dtype)
        np.copyto(arr, value, casting='unsafe')
    yield copyto_scalar

    def copyto():
        arr = np.empty(3, dtype=dtype)
        np.copyto(arr, np.array([value, value, value]), casting='unsafe')
    yield copyto

    def copyto_scalar_masked():
        arr = np.empty(3, dtype=dtype)
        np.copyto(arr, value, casting='unsafe', where=[True, False, True])
    yield copyto_scalar_masked

    def copyto_masked():
        arr = np.empty(3, dtype=dtype)
        np.copyto(arr, np.array([value, value, value]), casting='unsafe', where=[True, False, True])
    yield copyto_masked

    def direct_cast():
        np.array([value, value, value]).astype(dtype)
    yield direct_cast

    def direct_cast_nd_strided():
        arr = np.full((5, 5, 5), fill_value=value)[:, ::2, :]
        arr.astype(dtype)
    yield direct_cast_nd_strided

    def boolean_array_assignment():
        arr = np.empty(3, dtype=dtype)
        arr[[True, False, True]] = np.array([value, value])
    yield boolean_array_assignment

    def integer_array_assignment():
        arr = np.empty(3, dtype=dtype)
        values = np.array([value, value])
        arr[[0, 1]] = values
    yield integer_array_assignment

    def integer_array_assignment_with_subspace():
        arr = np.empty((5, 3), dtype=dtype)
        values = np.array([value, value, value])
        arr[[0, 2]] = values
    yield integer_array_assignment_with_subspace

    def flat_assignment():
        arr = np.empty((3,), dtype=dtype)
        values = np.array([value, value, value])
        arr.flat[:] = values
    yield flat_assignment