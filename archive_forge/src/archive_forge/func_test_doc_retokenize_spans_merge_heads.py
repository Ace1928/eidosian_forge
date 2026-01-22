import pytest
from spacy.attrs import LEMMA
from spacy.tokens import Doc, Token
from spacy.vocab import Vocab
def test_doc_retokenize_spans_merge_heads(en_vocab):
    words = ['I', 'found', 'a', 'pilates', 'class', 'near', 'work', '.']
    heads = [1, 1, 4, 6, 1, 4, 5, 1]
    deps = ['dep'] * len(heads)
    doc = Doc(en_vocab, words=words, heads=heads, deps=deps)
    assert len(doc) == 8
    with doc.retokenize() as retokenizer:
        attrs = {'tag': doc[4].tag_, 'lemma': 'pilates class', 'ent_type': 'O'}
        retokenizer.merge(doc[3:5], attrs=attrs)
    assert len(doc) == 7
    assert doc[0].head.i == 1
    assert doc[1].head.i == 1
    assert doc[2].head.i == 3
    assert doc[3].head.i == 1
    assert doc[4].head.i in [1, 3]
    assert doc[5].head.i == 4