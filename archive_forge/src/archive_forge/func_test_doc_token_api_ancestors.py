import numpy
import pytest
from spacy.attrs import IS_ALPHA, IS_DIGIT, IS_LOWER, IS_PUNCT, IS_STOP, IS_TITLE
from spacy.symbols import VERB
from spacy.tokens import Doc
from spacy.training import Example
from spacy.vocab import Vocab
def test_doc_token_api_ancestors(en_vocab):
    words = ['Yesterday', 'I', 'saw', 'a', 'dog', 'that', 'barked', 'loudly', '.']
    heads = [2, 2, 2, 4, 2, 6, 4, 6, 2]
    deps = ['dep'] * len(heads)
    doc = Doc(en_vocab, words=words, heads=heads, deps=deps)
    assert [t.text for t in doc[6].ancestors] == ['dog', 'saw']
    assert [t.text for t in doc[1].ancestors] == ['saw']
    assert [t.text for t in doc[2].ancestors] == []
    assert doc[2].is_ancestor(doc[7])
    assert not doc[6].is_ancestor(doc[2])