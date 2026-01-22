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
def get_characters_loss(ops, docs, prediction, nr_char):
    """Compute a loss based on a number of characters predicted from the docs."""
    target_ids = numpy.vstack([doc.to_utf8_array(nr_char=nr_char) for doc in docs])
    target_ids = target_ids.reshape((-1,))
    target = ops.asarray(to_categorical(target_ids, n_classes=256), dtype='f')
    target = target.reshape((-1, 256 * nr_char))
    diff = prediction - target
    loss = (diff ** 2).sum()
    d_target = diff / float(prediction.shape[0])
    return (loss, d_target)