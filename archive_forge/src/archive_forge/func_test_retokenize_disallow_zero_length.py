import pytest
from spacy.attrs import LEMMA
from spacy.tokens import Doc, Token
from spacy.vocab import Vocab
def test_retokenize_disallow_zero_length(en_vocab):
    doc = Doc(en_vocab, words=['hello', 'world', '!'])
    with pytest.raises(ValueError):
        with doc.retokenize() as retokenizer:
            retokenizer.merge(doc[1:1])