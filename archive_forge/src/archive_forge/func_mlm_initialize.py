from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Iterable, List, Optional, Tuple, cast
import numpy
from thinc.api import (
from thinc.loss import Loss
from thinc.types import Floats2d, Ints1d
from ...attrs import ID, ORTH
from ...errors import Errors
from ...util import OOV_RANK, registry
from ...vectors import Mode as VectorsMode
def mlm_initialize(model: Model, X=None, Y=None):
    wrapped = model.layers[0]
    wrapped.initialize(X=X, Y=Y)
    for dim in wrapped.dim_names:
        if wrapped.has_dim(dim):
            model.set_dim(dim, wrapped.get_dim(dim))