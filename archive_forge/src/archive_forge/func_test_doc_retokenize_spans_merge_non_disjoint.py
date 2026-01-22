import pytest
from spacy.attrs import LEMMA
from spacy.tokens import Doc, Token
from spacy.vocab import Vocab
def test_doc_retokenize_spans_merge_non_disjoint(en_tokenizer):
    text = 'Los Angeles start.'
    doc = en_tokenizer(text)
    with pytest.raises(ValueError):
        with doc.retokenize() as retokenizer:
            retokenizer.merge(doc[0:2], attrs={'tag': 'NNP', 'lemma': 'Los Angeles', 'ent_type': 'GPE'})
            retokenizer.merge(doc[0:1], attrs={'tag': 'NNP', 'lemma': 'Los Angeles', 'ent_type': 'GPE'})