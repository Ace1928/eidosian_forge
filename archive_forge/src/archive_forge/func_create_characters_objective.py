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
def create_characters_objective(vocab: 'Vocab', tok2vec: Model) -> Model:
    model = build_cloze_characters_multi_task_model(vocab, tok2vec, hidden_size=hidden_size, maxout_pieces=maxout_pieces, nr_char=n_characters)
    model.attrs['loss'] = partial(get_characters_loss, nr_char=n_characters)
    return model