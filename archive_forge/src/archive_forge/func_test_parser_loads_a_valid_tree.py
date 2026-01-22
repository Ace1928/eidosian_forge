import pytest
from nltk.data import find
from nltk.parse.bllip import BllipParser
from nltk.tree import Tree
def test_parser_loads_a_valid_tree(self, parser):
    parsed = parser.parse('I saw the man with the telescope')
    tree = next(parsed)
    assert isinstance(tree, Tree)
    assert tree.pformat() == '\n(S1\n  (S\n    (NP (PRP I))\n    (VP\n      (VBD saw)\n      (NP (DT the) (NN man))\n      (PP (IN with) (NP (DT the) (NN telescope))))))\n'.strip()