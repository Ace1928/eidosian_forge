import numpy
import pytest
from numpy.testing import assert_array_equal
from thinc.api import get_current_ops
from spacy.attrs import LENGTH, ORTH
from spacy.lang.en import English
from spacy.tokens import Doc, Span, Token
from spacy.util import filter_spans
from spacy.vocab import Vocab
from ..util import add_vecs_to_vocab
from .test_underscore import clean_underscore  # noqa: F401
def test_for_partial_ent_sents():
    """Spans may be associated with multiple sentences. These .sents should always be complete, not partial, sentences,
    which this tests for.
    """
    doc = Doc(English().vocab, words=["Mahler's", 'Symphony', 'No.', '8', 'was', 'beautiful.'], sent_starts=[1, 0, 0, 1, 0, 0])
    doc.set_ents([Span(doc, 1, 4, 'WORK')])
    for doc_sent, ent_sent in zip(doc.sents, doc.ents[0].sents):
        assert doc_sent == ent_sent