import pytest
from numpy.testing import assert_equal
from thinc.api import Adam
from spacy import registry, util
from spacy.attrs import DEP, NORM
from spacy.lang.en import English
from spacy.pipeline import DependencyParser
from spacy.pipeline.dep_parser import DEFAULT_PARSER_MODEL
from spacy.pipeline.tok2vec import DEFAULT_TOK2VEC_MODEL
from spacy.tokens import Doc
from spacy.training import Example
from spacy.vocab import Vocab
from ..util import apply_transition_sequence, make_tempdir
@pytest.mark.issue(7716)
@pytest.mark.xfail(reason='Not fixed yet')
def test_partial_annotation(parser):
    doc = Doc(parser.vocab, words=['a', 'b', 'c', 'd'])
    doc[2].is_sent_start = False
    doc = parser(doc)
    assert doc[2].is_sent_start == False