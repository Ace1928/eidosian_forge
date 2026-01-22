import numpy
import pytest
from spacy.tokens import Doc, Token
from spacy.vocab import Vocab
def test_doc_retokenizer_split_norm(en_vocab):
    """#6060: reset norm in split"""
    text = 'The quick brownfoxjumpsoverthe lazy dog w/ white spots'
    doc = Doc(en_vocab, words=text.split())
    doc[5].norm_ = 'with'
    token = doc[2]
    with doc.retokenize() as retokenizer:
        retokenizer.split(token, ['brown', 'fox', 'jumps', 'over', 'the'], heads=[(token, idx) for idx in range(5)])
    assert doc[9].text == 'w/'
    assert doc[9].norm_ == 'with'
    assert doc[5].text == 'over'
    assert doc[5].norm_ == 'over'