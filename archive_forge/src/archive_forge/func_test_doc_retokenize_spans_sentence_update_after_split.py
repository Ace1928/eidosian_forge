import numpy
import pytest
from spacy.tokens import Doc, Token
from spacy.vocab import Vocab
def test_doc_retokenize_spans_sentence_update_after_split(en_vocab):
    words = ['StewartLee', 'is', 'a', 'stand', 'up', 'comedian', '.', 'He', 'lives', 'in', 'England', 'and', 'loves', 'JoePasquale', '.']
    heads = [1, 1, 3, 5, 3, 1, 1, 8, 8, 8, 9, 8, 8, 14, 12]
    deps = ['nsubj', 'ROOT', 'det', 'amod', 'prt', 'attr', 'punct', 'nsubj', 'ROOT', 'prep', 'pobj', 'cc', 'conj', 'compound', 'punct']
    doc = Doc(en_vocab, words=words, heads=heads, deps=deps)
    sent1, sent2 = list(doc.sents)
    init_len = len(sent1)
    init_len2 = len(sent2)
    with doc.retokenize() as retokenizer:
        retokenizer.split(doc[0], ['Stewart', 'Lee'], [(doc[0], 1), doc[1]], attrs={'dep': ['compound', 'nsubj']})
        retokenizer.split(doc[13], ['Joe', 'Pasquale'], [(doc[13], 1), doc[12]], attrs={'dep': ['compound', 'dobj']})
    sent1, sent2 = list(doc.sents)
    assert len(sent1) == init_len + 1
    assert len(sent2) == init_len2 + 1