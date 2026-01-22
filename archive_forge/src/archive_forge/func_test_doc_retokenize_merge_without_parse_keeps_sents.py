import pytest
from spacy.attrs import LEMMA
from spacy.tokens import Doc, Token
from spacy.vocab import Vocab
def test_doc_retokenize_merge_without_parse_keeps_sents(en_tokenizer):
    text = 'displaCy is a parse tool built with Javascript'
    sent_starts = [1, 0, 0, 0, 1, 0, 0, 0]
    tokens = en_tokenizer(text)
    doc = Doc(tokens.vocab, words=[t.text for t in tokens], sent_starts=sent_starts)
    assert len(list(doc.sents)) == 2
    with doc.retokenize() as retokenizer:
        retokenizer.merge(doc[1:3])
    assert len(list(doc.sents)) == 2
    doc = Doc(tokens.vocab, words=[t.text for t in tokens], sent_starts=sent_starts)
    assert len(list(doc.sents)) == 2
    with doc.retokenize() as retokenizer:
        retokenizer.merge(doc[3:6])
    assert doc[3].is_sent_start is None
    doc = Doc(tokens.vocab, words=[t.text for t in tokens], sent_starts=sent_starts)
    assert len(list(doc.sents)) == 2
    with doc.retokenize() as retokenizer:
        retokenizer.merge(doc[3:6], attrs={'sent_start': True})
    assert len(list(doc.sents)) == 2