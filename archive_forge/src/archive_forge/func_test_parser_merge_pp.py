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
def test_parser_merge_pp(en_vocab):
    words = ['A', 'phrase', 'with', 'another', 'phrase', 'occurs']
    heads = [1, 5, 1, 4, 2, 5]
    deps = ['det', 'nsubj', 'prep', 'det', 'pobj', 'ROOT']
    pos = ['DET', 'NOUN', 'ADP', 'DET', 'NOUN', 'VERB']
    doc = Doc(en_vocab, words=words, deps=deps, heads=heads, pos=pos)
    with doc.retokenize() as retokenizer:
        for np in doc.noun_chunks:
            retokenizer.merge(np, attrs={'lemma': np.lemma_})
    assert doc[0].text == 'A phrase'
    assert doc[1].text == 'with'
    assert doc[2].text == 'another phrase'
    assert doc[3].text == 'occurs'