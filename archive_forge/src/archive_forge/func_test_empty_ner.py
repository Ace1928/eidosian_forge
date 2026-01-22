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
def test_empty_ner():
    nlp = English()
    ner = nlp.add_pipe('ner')
    ner.add_label('MY_LABEL')
    nlp.initialize()
    doc = nlp("John is watching the news about Croatia's elections")
    result = ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
    assert [token.ent_iob_ for token in doc] == result