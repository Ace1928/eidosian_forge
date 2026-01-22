import pytest
from spacy.attrs import LEMMA
from spacy.tokens import Doc, Token
from spacy.vocab import Vocab
def test_doc_retokenize_retokenizer(en_tokenizer):
    doc = en_tokenizer('WKRO played songs by the beach boys all night')
    with doc.retokenize() as retokenizer:
        retokenizer.merge(doc[4:7])
    assert len(doc) == 7
    assert doc[4].text == 'the beach boys'