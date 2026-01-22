import numpy
import pytest
from spacy.attrs import IS_ALPHA, IS_DIGIT, IS_LOWER, IS_PUNCT, IS_STOP, IS_TITLE
from spacy.symbols import VERB
from spacy.tokens import Doc
from spacy.training import Example
from spacy.vocab import Vocab
def test_token_api_conjuncts_chain(en_vocab):
    words = ['The', 'boy', 'and', 'the', 'girl', 'and', 'the', 'man', 'went', '.']
    heads = [1, 8, 1, 4, 1, 4, 7, 4, 8, 8]
    deps = ['det', 'nsubj', 'cc', 'det', 'conj', 'cc', 'det', 'conj', 'ROOT', 'punct']
    doc = Doc(en_vocab, words=words, heads=heads, deps=deps)
    assert [w.text for w in doc[1].conjuncts] == ['girl', 'man']
    assert [w.text for w in doc[4].conjuncts] == ['boy', 'man']
    assert [w.text for w in doc[7].conjuncts] == ['boy', 'girl']