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
def test_ner_constructor(en_vocab):
    config = {'update_with_oracle_cut_size': 100}
    cfg = {'model': DEFAULT_NER_MODEL}
    model = registry.resolve(cfg, validate=True)['model']
    EntityRecognizer(en_vocab, model, **config)
    EntityRecognizer(en_vocab, model)