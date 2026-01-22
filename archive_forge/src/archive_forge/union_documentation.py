from functools import partial
import numpy as np
from traitlets import Union, Instance, Undefined, TraitError
from .serializers import data_union_serialization
from .traits import NDArray
from .widgets import NDArrayWidget, NDArrayBase, NDArraySource

    Union trait of NDArray and NDArrayBase for numpy arrays.
    