import logging
import random
import pytest
from numpy.testing import assert_equal
from spacy import registry, util
from spacy.attrs import ENT_IOB
from spacy.lang.en import English
from spacy.lang.it import Italian
from spacy.language import Language
from spacy.lookups import Lookups
from spacy.pipeline import EntityRecognizer
from spacy.pipeline._parser_internals.ner import BiluoPushDown
from spacy.pipeline.ner import DEFAULT_NER_MODEL
from spacy.tokens import Doc, Span
from spacy.training import Example, iob_to_biluo, split_bilu_label
from spacy.vocab import Vocab
from ..util import make_tempdir
def test_beam_valid_parse(neg_key):
    """Regression test for previously flakey behaviour"""
    nlp = English()
    beam_width = 16
    beam_density = 0.0001
    config = {'beam_width': beam_width, 'beam_density': beam_density, 'incorrect_spans_key': neg_key}
    nlp.add_pipe('beam_ner', config=config)
    tokens = ['FEDERAL', 'NATIONAL', 'MORTGAGE', 'ASSOCIATION', '(', 'Fannie', 'Mae', '):', 'Posted', 'yields', 'on', '30', 'year', 'mortgage', 'commitments', 'for', 'delivery', 'within', '30', 'days', '(', 'priced', 'at', 'par', ')', '9.75', '%', ',', 'standard', 'conventional', 'fixed', '-', 'rate', 'mortgages', ';', '8.70', '%', ',', '6/2', 'rate', 'capped', 'one', '-', 'year', 'adjustable', 'rate', 'mortgages', '.', 'Source', ':', 'Telerate', 'Systems', 'Inc.']
    iob = ['B-ORG', 'I-ORG', 'I-ORG', 'L-ORG', 'O', 'B-ORG', 'L-ORG', 'O', 'O', 'O', 'O', 'B-DATE', 'L-DATE', 'O', 'O', 'O', 'O', 'O', 'B-DATE', 'L-DATE', 'O', 'O', 'O', 'O', 'O', 'B-PERCENT', 'L-PERCENT', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-PERCENT', 'L-PERCENT', 'O', 'U-CARDINAL', 'O', 'O', 'B-DATE', 'I-DATE', 'L-DATE', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
    doc = Doc(nlp.vocab, words=tokens)
    example = Example.from_dict(doc, {'ner': iob})
    neg_span = Span(example.reference, 50, 53, 'ORG')
    example.reference.spans[neg_key] = [neg_span]
    optimizer = nlp.initialize()
    for i in range(5):
        losses = {}
        nlp.update([example], sgd=optimizer, losses=losses)
    assert 'beam_ner' in losses