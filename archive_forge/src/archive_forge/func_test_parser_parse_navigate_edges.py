import pytest
from spacy.tokens import Doc
def test_parser_parse_navigate_edges(en_vocab, words, heads):
    doc = Doc(en_vocab, words=words, heads=heads, deps=['dep'] * len(heads))
    for token in doc:
        subtree = list(token.subtree)
        debug = '\t'.join((token.text, token.left_edge.text, subtree[0].text))
        assert token.left_edge == subtree[0], debug
        debug = '\t'.join((token.text, token.right_edge.text, subtree[-1].text, token.right_edge.head.text))
        assert token.right_edge == subtree[-1], debug