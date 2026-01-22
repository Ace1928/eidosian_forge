from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union, cast
import numpy
from thinc.api import Config, Model, Ops, Optimizer, get_current_ops, set_dropout_rate
from thinc.types import Floats2d, Ints1d, Ints2d, Ragged
from ..compat import Protocol, runtime_checkable
from ..errors import Errors
from ..language import Language
from ..scorer import Scorer
from ..tokens import Doc, Span, SpanGroup
from ..training import Example, validate_examples
from ..util import registry
from ..vocab import Vocab
from .trainable_pipe import TrainablePipe
def ngram_suggester(docs: Iterable[Doc], sizes: List[int], *, ops: Optional[Ops]=None) -> Ragged:
    if ops is None:
        ops = get_current_ops()
    spans = []
    lengths = []
    for doc in docs:
        starts = ops.xp.arange(len(doc), dtype='i')
        starts = starts.reshape((-1, 1))
        length = 0
        for size in sizes:
            if size <= len(doc):
                starts_size = starts[:len(doc) - (size - 1)]
                spans.append(ops.xp.hstack((starts_size, starts_size + size)))
                length += spans[-1].shape[0]
            if spans:
                assert spans[-1].ndim == 2, spans[-1].shape
        lengths.append(length)
    lengths_array = ops.asarray1i(lengths)
    if len(spans) > 0:
        output = Ragged(ops.xp.vstack(spans), lengths_array)
    else:
        output = Ragged(ops.xp.zeros((0, 0), dtype='i'), lengths_array)
    assert output.dataXd.ndim == 2
    return output