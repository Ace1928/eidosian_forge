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
def build_multi_task_model(tok2vec: Model, maxout_pieces: int, token_vector_width: int, nO: Optional[int]=None) -> Model:
    softmax = Softmax(nO=nO, nI=token_vector_width * 2)
    model = chain(tok2vec, Maxout(nO=token_vector_width * 2, nI=token_vector_width, nP=maxout_pieces, dropout=0.0), LayerNorm(token_vector_width * 2), softmax)
    model.set_ref('tok2vec', tok2vec)
    model.set_ref('output_layer', softmax)
    return model