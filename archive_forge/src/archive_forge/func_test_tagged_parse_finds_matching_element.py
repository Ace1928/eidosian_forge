import pytest
from nltk.data import find
from nltk.parse.bllip import BllipParser
from nltk.tree import Tree
def test_tagged_parse_finds_matching_element(self, parser):
    parsed = parser.parse('I saw the man with the telescope')
    tagged_tree = next(parser.tagged_parse([('telescope', 'NN')]))
    assert isinstance(tagged_tree, Tree)
    assert tagged_tree.pformat() == '(S1 (NP (NN telescope)))'