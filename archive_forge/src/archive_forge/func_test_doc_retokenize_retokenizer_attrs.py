import pytest
from spacy.attrs import LEMMA
from spacy.tokens import Doc, Token
from spacy.vocab import Vocab
def test_doc_retokenize_retokenizer_attrs(en_tokenizer):
    doc = en_tokenizer('WKRO played songs by the beach boys all night')
    attrs = {LEMMA: 'boys', 'ENT_TYPE': doc.vocab.strings['ORG']}
    with doc.retokenize() as retokenizer:
        retokenizer.merge(doc[4:7], attrs=attrs)
    assert len(doc) == 7
    assert doc[4].text == 'the beach boys'
    assert doc[4].lemma_ == 'boys'
    assert doc[4].ent_type_ == 'ORG'