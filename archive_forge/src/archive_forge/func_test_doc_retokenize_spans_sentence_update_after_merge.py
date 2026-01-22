import pytest
from spacy.attrs import LEMMA
from spacy.tokens import Doc, Token
from spacy.vocab import Vocab
def test_doc_retokenize_spans_sentence_update_after_merge(en_tokenizer):
    text = 'Stewart Lee is a stand up comedian. He lives in England and loves Joe Pasquale.'
    heads = [1, 2, 2, 4, 2, 4, 4, 2, 9, 9, 9, 10, 9, 9, 15, 13, 9]
    deps = ['compound', 'nsubj', 'ROOT', 'det', 'amod', 'prt', 'attr', 'punct', 'nsubj', 'ROOT', 'prep', 'pobj', 'cc', 'conj', 'compound', 'dobj', 'punct']
    tokens = en_tokenizer(text)
    doc = Doc(tokens.vocab, words=[t.text for t in tokens], heads=heads, deps=deps)
    sent1, sent2 = list(doc.sents)
    init_len = len(sent1)
    init_len2 = len(sent2)
    with doc.retokenize() as retokenizer:
        attrs = {'lemma': 'none', 'ent_type': 'none'}
        retokenizer.merge(doc[0:2], attrs=attrs)
        retokenizer.merge(doc[-2:], attrs=attrs)
    sent1, sent2 = list(doc.sents)
    assert len(sent1) == init_len - 1
    assert len(sent2) == init_len2 - 1