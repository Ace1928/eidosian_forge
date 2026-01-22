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
def create_vectors_loss() -> Callable:
    distance: Loss
    if loss == 'cosine':
        distance = CosineDistance(normalize=True, ignore_zeros=True)
        return partial(get_vectors_loss, distance=distance)
    elif loss == 'L2':
        distance = L2Distance(normalize=True)
        return partial(get_vectors_loss, distance=distance)
    else:
        raise ValueError(Errors.E906.format(found=loss, supported="'cosine', 'L2'"))