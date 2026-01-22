import numpy
import pytest
from spacy.attrs import IS_ALPHA, IS_DIGIT, IS_LOWER, IS_PUNCT, IS_STOP, IS_TITLE
from spacy.symbols import VERB
from spacy.tokens import Doc
from spacy.training import Example
from spacy.vocab import Vocab
def test_doc_token_api_head_setter(en_vocab):
    words = ['Yesterday', 'I', 'saw', 'a', 'dog', 'that', 'barked', 'loudly', '.']
    heads = [2, 2, 2, 4, 2, 6, 4, 6, 2]
    deps = ['dep'] * len(heads)
    doc = Doc(en_vocab, words=words, heads=heads, deps=deps)
    assert doc[6].n_lefts == 1
    assert doc[6].n_rights == 1
    assert doc[6].left_edge.i == 5
    assert doc[6].right_edge.i == 7
    assert doc[4].n_lefts == 1
    assert doc[4].n_rights == 1
    assert doc[4].left_edge.i == 3
    assert doc[4].right_edge.i == 7
    assert doc[3].n_lefts == 0
    assert doc[3].n_rights == 0
    assert doc[3].left_edge.i == 3
    assert doc[3].right_edge.i == 3
    assert doc[2].left_edge.i == 0
    assert doc[2].right_edge.i == 8
    doc[6].head = doc[3]
    assert doc[6].n_lefts == 1
    assert doc[6].n_rights == 1
    assert doc[6].left_edge.i == 5
    assert doc[6].right_edge.i == 7
    assert doc[3].n_lefts == 0
    assert doc[3].n_rights == 1
    assert doc[3].left_edge.i == 3
    assert doc[3].right_edge.i == 7
    assert doc[4].n_lefts == 1
    assert doc[4].n_rights == 0
    assert doc[4].left_edge.i == 3
    assert doc[4].right_edge.i == 7
    assert doc[2].left_edge.i == 0
    assert doc[2].right_edge.i == 8
    doc[0].head = doc[5]
    assert doc[5].left_edge.i == 0
    assert doc[6].left_edge.i == 0
    assert doc[3].left_edge.i == 0
    assert doc[4].left_edge.i == 0
    assert doc[2].left_edge.i == 0
    doc2 = Doc(en_vocab, words=words, heads=heads, deps=['dep'] * len(heads))
    with pytest.raises(ValueError):
        doc[0].head = doc2[0]
    words = ['This', 'is', 'one', 'sentence', '.', 'This', 'is', 'another', 'sentence', '.']
    heads = [0, 0, 0, 0, 0, 5, 5, 5, 5, 5]
    doc = Doc(en_vocab, words=words, heads=heads, deps=['dep'] * len(heads))
    assert doc[0].is_sent_start
    assert doc[5].is_sent_start
    assert doc[0].left_edge == doc[0]
    assert doc[0].right_edge == doc[4]
    assert doc[5].left_edge == doc[5]
    assert doc[5].right_edge == doc[9]
    doc[2].head = doc[3]
    assert doc[0].is_sent_start
    assert doc[5].is_sent_start
    assert doc[0].left_edge == doc[0]
    assert doc[0].right_edge == doc[4]
    assert doc[5].left_edge == doc[5]
    assert doc[5].right_edge == doc[9]
    doc[5].head = doc[0]
    assert doc[0].is_sent_start
    assert not doc[5].is_sent_start
    assert doc[0].left_edge == doc[0]
    assert doc[0].right_edge == doc[9]