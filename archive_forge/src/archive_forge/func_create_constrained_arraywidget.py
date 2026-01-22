from contextlib import contextmanager
from ipywidgets import register
from traitlets import Unicode, Set, Undefined, Int, validate
import numpy as np
from ..widgets import DataWidget
from .traits import NDArray
from .serializers import compressed_array_serialization
from inspect import Signature, Parameter
def create_constrained_arraywidget(*validators, **kwargs):
    """Returns a subclass of NDArrayWidget with a constrained array.

    Accepts keyword argument 'dtype' in addition to any valdiators.
    """
    dtype = kwargs.pop('dtype', None)
    return type('ConstrainedNDArrayWidget', (NDArrayWidget,), {'array': NDArray(dtype=dtype).tag(sync=True, **compressed_array_serialization).valid(*validators)})