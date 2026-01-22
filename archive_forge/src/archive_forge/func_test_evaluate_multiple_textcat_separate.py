import itertools
import logging
import warnings
from unittest import mock
import pytest
from thinc.api import CupyOps, NumpyOps, get_current_ops
import spacy
from spacy.lang.de import German
from spacy.lang.en import English
from spacy.language import Language
from spacy.scorer import Scorer
from spacy.tokens import Doc, Span
from spacy.training import Example
from spacy.util import find_matching_language, ignore_error, raise_error, registry
from spacy.vocab import Vocab
from .util import add_vecs_to_vocab, assert_docs_equal
def test_evaluate_multiple_textcat_separate(en_vocab):
    """Test that evaluate can evaluate multiple textcat components separately
    with custom scorers."""

    def custom_textcat_score(examples, **kwargs):
        scores = Scorer.score_cats(examples, 'cats', multi_label=False, **kwargs)
        return {f'custom_{k}': v for k, v in scores.items()}

    @spacy.registry.scorers('test_custom_textcat_scorer')
    def make_custom_textcat_scorer():
        return custom_textcat_score
    nlp = Language(en_vocab)
    textcat = nlp.add_pipe('textcat', config={'scorer': {'@scorers': 'test_custom_textcat_scorer'}})
    for label in ('POSITIVE', 'NEGATIVE'):
        textcat.add_label(label)
    textcat_multilabel = nlp.add_pipe('textcat_multilabel')
    for label in ('FEATURE', 'REQUEST', 'BUG', 'QUESTION'):
        textcat_multilabel.add_label(label)
    nlp.initialize()
    annots = {'cats': {'POSITIVE': 1.0, 'NEGATIVE': 0.0, 'FEATURE': 1.0, 'QUESTION': 1.0, 'POSITIVE': 1.0, 'NEGATIVE': 0.0}}
    doc = nlp.make_doc('hello world')
    example = Example.from_dict(doc, annots)
    scores = nlp.evaluate([example])
    assert 'custom_cats_f_per_type' in scores
    labels = nlp.get_pipe('textcat').labels
    assert set(scores['custom_cats_f_per_type'].keys()) == set(labels)
    assert 'cats_f_per_type' in scores
    labels = nlp.get_pipe('textcat_multilabel').labels
    assert set(scores['cats_f_per_type'].keys()) == set(labels)