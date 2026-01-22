import pytest
from spacy.tokens import Doc
from ..util import apply_transition_sequence
def test_parser_sentence_space(en_vocab):
    words = ['I', 'look', 'forward', 'to', 'using', 'Thingamajig', '.', ' ', 'I', "'ve", 'been', 'told', 'it', 'will', 'make', 'my', 'life', 'easier', '...']
    heads = [1, 1, 1, 1, 3, 4, 1, 6, 11, 11, 11, 11, 14, 14, 11, 16, 17, 14, 11]
    deps = ['nsubj', 'ROOT', 'advmod', 'prep', 'pcomp', 'dobj', 'punct', '', 'nsubjpass', 'aux', 'auxpass', 'ROOT', 'nsubj', 'aux', 'ccomp', 'poss', 'nsubj', 'ccomp', 'punct']
    doc = Doc(en_vocab, words=words, heads=heads, deps=deps)
    assert len(list(doc.sents)) == 2