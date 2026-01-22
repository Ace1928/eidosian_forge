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
def test_neg_annotation(neg_key):
    """Check that the NER update works with a negative annotation that is a different label of the correct one,
    or partly overlapping, etc"""
    nlp = English()
    beam_width = 16
    beam_density = 0.0001
    config = {'beam_width': beam_width, 'beam_density': beam_density, 'incorrect_spans_key': neg_key}
    ner = nlp.add_pipe('beam_ner', config=config)
    train_text = 'Who is Shaka Khan?'
    neg_doc = nlp.make_doc(train_text)
    ner.add_label('PERSON')
    ner.add_label('ORG')
    example = Example.from_dict(neg_doc, {'entities': [(7, 17, 'PERSON')]})
    example.reference.spans[neg_key] = [Span(example.reference, 2, 4, 'ORG'), Span(example.reference, 2, 3, 'PERSON'), Span(example.reference, 1, 4, 'PERSON')]
    optimizer = nlp.initialize()
    for i in range(2):
        losses = {}
        nlp.update([example], sgd=optimizer, losses=losses)