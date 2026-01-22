import pytest
from spacy.pipeline._parser_internals import nonproj
from spacy.pipeline._parser_internals.nonproj import (
from spacy.tokens import Doc
def test_parser_is_nonproj_arc(cyclic_tree, nonproj_tree, partial_tree, multirooted_tree):
    assert is_nonproj_arc(0, nonproj_tree) is False
    assert is_nonproj_arc(1, nonproj_tree) is False
    assert is_nonproj_arc(2, nonproj_tree) is False
    assert is_nonproj_arc(3, nonproj_tree) is False
    assert is_nonproj_arc(4, nonproj_tree) is False
    assert is_nonproj_arc(5, nonproj_tree) is False
    assert is_nonproj_arc(6, nonproj_tree) is False
    assert is_nonproj_arc(7, nonproj_tree) is True
    assert is_nonproj_arc(8, nonproj_tree) is False
    assert is_nonproj_arc(7, partial_tree) is False
    assert is_nonproj_arc(17, multirooted_tree) is False
    assert is_nonproj_arc(16, multirooted_tree) is True
    with pytest.raises(ValueError, match='Found cycle in dependency graph: \\[1, 2, 2, 4, 5, 3, 2\\]'):
        is_nonproj_arc(6, cyclic_tree)