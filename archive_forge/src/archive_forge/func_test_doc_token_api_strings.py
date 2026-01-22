import numpy
import pytest
from spacy.attrs import IS_ALPHA, IS_DIGIT, IS_LOWER, IS_PUNCT, IS_STOP, IS_TITLE
from spacy.symbols import VERB
from spacy.tokens import Doc
from spacy.training import Example
from spacy.vocab import Vocab
def test_doc_token_api_strings(en_vocab):
    words = ['Give', 'it', 'back', '!', 'He', 'pleaded', '.']
    pos = ['VERB', 'PRON', 'PART', 'PUNCT', 'PRON', 'VERB', 'PUNCT']
    heads = [0, 0, 0, 0, 5, 5, 5]
    deps = ['ROOT', 'dobj', 'prt', 'punct', 'nsubj', 'ROOT', 'punct']
    doc = Doc(en_vocab, words=words, pos=pos, heads=heads, deps=deps)
    assert doc[0].orth_ == 'Give'
    assert doc[0].text == 'Give'
    assert doc[0].text_with_ws == 'Give '
    assert doc[0].lower_ == 'give'
    assert doc[0].shape_ == 'Xxxx'
    assert doc[0].prefix_ == 'G'
    assert doc[0].suffix_ == 'ive'
    assert doc[0].pos_ == 'VERB'
    assert doc[0].dep_ == 'ROOT'