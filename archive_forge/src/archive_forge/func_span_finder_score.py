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
def span_finder_score(examples: Iterable[Example], **kwargs) -> Dict[str, Any]:
    kwargs = dict(kwargs)
    attr_prefix = 'spans_'
    key = kwargs['spans_key']
    kwargs.setdefault('attr', f'{attr_prefix}{key}')
    kwargs.setdefault('getter', lambda doc, key: doc.spans.get(key[len(attr_prefix):], []))
    kwargs.setdefault('has_annotation', lambda doc: key in doc.spans)
    kwargs.setdefault('allow_overlap', True)
    kwargs.setdefault('labeled', False)
    scores = Scorer.score_spans(examples, **kwargs)
    scores.pop(f'{kwargs['attr']}_per_type', None)
    return scores