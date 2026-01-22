import pytest
from spacy.attrs import LEMMA
from spacy.tokens import Doc, Token
from spacy.vocab import Vocab
def test_doc_retokenize_merge_hang(en_tokenizer):
    text = 'through North and South Carolina'
    doc = en_tokenizer(text)
    with doc.retokenize() as retokenizer:
        retokenizer.merge(doc[3:5], attrs={'lemma': '', 'ent_type': 'ORG'})
        retokenizer.merge(doc[1:2], attrs={'lemma': '', 'ent_type': 'ORG'})