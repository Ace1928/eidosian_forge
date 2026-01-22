import pytest
from spacy.pipeline._parser_internals import nonproj
from spacy.pipeline._parser_internals.nonproj import (
from spacy.tokens import Doc
def test_parser_ancestors(tree, cyclic_tree, partial_tree, multirooted_tree):
    assert [a for a in ancestors(3, tree)] == [4, 5, 2]
    assert [a for a in ancestors(3, cyclic_tree)] == [4, 5, 3, 4, 5, 3, 4]
    assert [a for a in ancestors(3, partial_tree)] == [4, 5, None]
    assert [a for a in ancestors(17, multirooted_tree)] == []