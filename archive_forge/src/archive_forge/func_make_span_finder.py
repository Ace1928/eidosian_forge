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
@Language.factory('span_finder', assigns=['doc.spans'], default_config={'threshold': 0.5, 'model': DEFAULT_SPAN_FINDER_MODEL, 'spans_key': DEFAULT_SPANS_KEY, 'max_length': 25, 'min_length': None, 'scorer': {'@scorers': 'spacy.span_finder_scorer.v1'}}, default_score_weights={f'spans_{DEFAULT_SPANS_KEY}_f': 1.0, f'spans_{DEFAULT_SPANS_KEY}_p': 0.0, f'spans_{DEFAULT_SPANS_KEY}_r': 0.0})
def make_span_finder(nlp: Language, name: str, model: Model[Iterable[Doc], Floats2d], spans_key: str, threshold: float, max_length: Optional[int], min_length: Optional[int], scorer: Optional[Callable]) -> 'SpanFinder':
    """Create a SpanFinder component. The component predicts whether a token is
    the start or the end of a potential span.

    model (Model[List[Doc], Floats2d]): A model instance that
        is given a list of documents and predicts a probability for each token.
    spans_key (str): Key of the doc.spans dict to save the spans under. During
        initialization and training, the component will look for spans on the
        reference document under the same key.
    threshold (float): Minimum probability to consider a prediction positive.
    max_length (Optional[int]): Maximum length of the produced spans, defaults
        to None meaning unlimited length.
    min_length (Optional[int]): Minimum length of the produced spans, defaults
        to None meaning shortest span length is 1.
    scorer (Optional[Callable]): The scoring method. Defaults to
        Scorer.score_spans for the Doc.spans[spans_key] with overlapping
        spans allowed.
    """
    return SpanFinder(nlp, model=model, threshold=threshold, name=name, scorer=scorer, max_length=max_length, min_length=min_length, spans_key=spans_key)