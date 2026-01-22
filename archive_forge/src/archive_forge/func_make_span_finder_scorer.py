from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from thinc.api import Config, Model, Optimizer, set_dropout_rate
from thinc.types import Floats2d
from ..errors import Errors
from ..language import Language
from ..scorer import Scorer
from ..tokens import Doc, Span
from ..training import Example
from ..util import registry
from .spancat import DEFAULT_SPANS_KEY
from .trainable_pipe import TrainablePipe
@registry.scorers('spacy.span_finder_scorer.v1')
def make_span_finder_scorer():
    return span_finder_score