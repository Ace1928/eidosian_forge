import numpy
import pytest
from spacy.attrs import IS_ALPHA, IS_DIGIT, IS_LOWER, IS_PUNCT, IS_STOP, IS_TITLE
from spacy.symbols import VERB
from spacy.tokens import Doc
from spacy.training import Example
from spacy.vocab import Vocab
def test_token_api_non_conjuncts(en_vocab):
    words = ['They', 'came', '.']
    heads = [1, 1, 1]
    deps = ['nsubj', 'ROOT', 'punct']
    doc = Doc(en_vocab, words=words, heads=heads, deps=deps)
    assert [w.text for w in doc[0].conjuncts] == []
    assert [w.text for w in doc[1].conjuncts] == []