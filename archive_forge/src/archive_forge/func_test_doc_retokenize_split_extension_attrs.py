import numpy
import pytest
from spacy.tokens import Doc, Token
from spacy.vocab import Vocab
def test_doc_retokenize_split_extension_attrs(en_vocab):
    Token.set_extension('a', default=False, force=True)
    Token.set_extension('b', default='nothing', force=True)
    doc = Doc(en_vocab, words=['LosAngeles', 'start'])
    with doc.retokenize() as retokenizer:
        heads = [(doc[0], 1), doc[1]]
        underscore = [{'a': True, 'b': '1'}, {'b': '2'}]
        attrs = {'lemma': ['los', 'angeles'], '_': underscore}
        retokenizer.split(doc[0], ['Los', 'Angeles'], heads, attrs=attrs)
    assert doc[0].lemma_ == 'los'
    assert doc[0]._.a is True
    assert doc[0]._.b == '1'
    assert doc[1].lemma_ == 'angeles'
    assert doc[1]._.a is False
    assert doc[1]._.b == '2'