import warnings
from functools import partial
from pathlib import Path
from typing import (
import srsly
from .. import util
from ..errors import Errors, Warnings
from ..language import Language
from ..matcher import Matcher, PhraseMatcher
from ..matcher.levenshtein import levenshtein_compare
from ..scorer import Scorer
from ..tokens import Doc, Span
from ..training import Example
from ..util import SimpleFrozenList, ensure_path, registry
from .pipe import Pipe
@Language.factory('span_ruler', assigns=['doc.spans'], default_config={'spans_key': DEFAULT_SPANS_KEY, 'spans_filter': None, 'annotate_ents': False, 'ents_filter': {'@misc': 'spacy.first_longest_spans_filter.v1'}, 'phrase_matcher_attr': None, 'matcher_fuzzy_compare': {'@misc': 'spacy.levenshtein_compare.v1'}, 'validate': False, 'overwrite': True, 'scorer': {'@scorers': 'spacy.overlapping_labeled_spans_scorer.v1', 'spans_key': DEFAULT_SPANS_KEY}}, default_score_weights={f'spans_{DEFAULT_SPANS_KEY}_f': 1.0, f'spans_{DEFAULT_SPANS_KEY}_p': 0.0, f'spans_{DEFAULT_SPANS_KEY}_r': 0.0, f'spans_{DEFAULT_SPANS_KEY}_per_type': None})
def make_span_ruler(nlp: Language, name: str, spans_key: Optional[str], spans_filter: Optional[Callable[[Iterable[Span], Iterable[Span]], Iterable[Span]]], annotate_ents: bool, ents_filter: Callable[[Iterable[Span], Iterable[Span]], Iterable[Span]], phrase_matcher_attr: Optional[Union[int, str]], matcher_fuzzy_compare: Callable, validate: bool, overwrite: bool, scorer: Optional[Callable]):
    return SpanRuler(nlp, name, spans_key=spans_key, spans_filter=spans_filter, annotate_ents=annotate_ents, ents_filter=ents_filter, phrase_matcher_attr=phrase_matcher_attr, matcher_fuzzy_compare=matcher_fuzzy_compare, validate=validate, overwrite=overwrite, scorer=scorer)