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
def build_cloze_multi_task_model(vocab: 'Vocab', tok2vec: Model, maxout_pieces: int, hidden_size: int) -> Model:
    nO = vocab.vectors.shape[1]
    output_layer = chain(cast(Model[List['Floats2d'], Floats2d], list2array()), Maxout(nO=hidden_size, nI=tok2vec.get_dim('nO'), nP=maxout_pieces, normalize=True, dropout=0.0), Linear(nO=nO, nI=hidden_size, init_W=zero_init))
    model = chain(tok2vec, output_layer)
    model = build_masked_language_model(vocab, model)
    model.set_ref('tok2vec', tok2vec)
    model.set_ref('output_layer', output_layer)
    return model