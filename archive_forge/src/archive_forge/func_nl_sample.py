import pytest
from spacy.tokens import Doc
from spacy.util import filter_spans
@pytest.fixture
def nl_sample(nl_vocab):
    words = ['Haar', 'vriend', 'lacht', 'luid', '.', 'We', 'kregen', 'alweer', 'ruzie', 'toen', 'we', 'de', 'supermarkt', 'ingingen', '.', 'Aan', 'het', 'begin', 'van', 'de', 'supermarkt', 'is', 'al', 'het', 'fruit', 'en', 'de', 'groentes', '.', 'Uiteindelijk', 'hebben', 'we', 'dan', 'ook', 'geen', 'avondeten', 'gekocht', '.']
    heads = [1, 2, 2, 2, 2, 6, 6, 6, 6, 13, 13, 12, 13, 6, 6, 17, 17, 24, 20, 20, 17, 24, 24, 24, 24, 27, 27, 24, 24, 36, 36, 36, 36, 36, 35, 36, 36, 36]
    deps = ['nmod:poss', 'nsubj', 'ROOT', 'advmod', 'punct', 'nsubj', 'ROOT', 'advmod', 'obj', 'mark', 'nsubj', 'det', 'obj', 'advcl', 'punct', 'case', 'det', 'obl', 'case', 'det', 'nmod', 'cop', 'advmod', 'det', 'ROOT', 'cc', 'det', 'conj', 'punct', 'advmod', 'aux', 'nsubj', 'advmod', 'advmod', 'det', 'obj', 'ROOT', 'punct']
    pos = ['PRON', 'NOUN', 'VERB', 'ADJ', 'PUNCT', 'PRON', 'VERB', 'ADV', 'NOUN', 'SCONJ', 'PRON', 'DET', 'NOUN', 'NOUN', 'PUNCT', 'ADP', 'DET', 'NOUN', 'ADP', 'DET', 'NOUN', 'AUX', 'ADV', 'DET', 'NOUN', 'CCONJ', 'DET', 'NOUN', 'PUNCT', 'ADJ', 'AUX', 'PRON', 'ADV', 'ADV', 'DET', 'NOUN', 'VERB', 'PUNCT']
    return Doc(nl_vocab, words=words, heads=heads, deps=deps, pos=pos)