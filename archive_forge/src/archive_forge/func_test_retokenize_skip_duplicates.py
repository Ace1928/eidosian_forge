import pytest
from spacy.attrs import LEMMA
from spacy.tokens import Doc, Token
from spacy.vocab import Vocab
def test_retokenize_skip_duplicates(en_vocab):
    """Test that the retokenizer automatically skips duplicate spans instead
    of complaining about overlaps. See #3687."""
    doc = Doc(en_vocab, words=['hello', 'world', '!'])
    with doc.retokenize() as retokenizer:
        retokenizer.merge(doc[0:2])
        retokenizer.merge(doc[0:2])
    assert len(doc) == 2
    assert doc[0].text == 'hello world'