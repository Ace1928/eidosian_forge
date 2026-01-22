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
def test_parser_root(en_vocab):
    words = ['i', 'do', "n't", 'have', 'other', 'assistance']
    heads = [3, 3, 3, 3, 5, 3]
    deps = ['nsubj', 'aux', 'neg', 'ROOT', 'amod', 'dobj']
    doc = Doc(en_vocab, words=words, heads=heads, deps=deps)
    for t in doc:
        assert t.dep != 0, t.text