import functools
from collections.abc import Sequence
import autoray as ar
import numpy as onp
from autograd.numpy.numpy_boxes import ArrayBox
from autoray import numpy as np
from numpy import ndarray
from . import single_dispatch  # pylint:disable=unused-import
from .utils import cast, cast_like, get_interface, requires_grad
Set the value at a specified index in an array.
    Calls ``array[idx]=val`` and returns the updated array unless JAX.

    Args:
        array (tensor_like): array to be modified
        idx (int, tuple): index to modify
        val (int, float): value to set

    Returns:
        a new copy of the array with the specified index updated to ``val``.

    Whether the original array is modified is interface-dependent.

    .. note:: TensorFlow EagerTensor does not support item assignment
    