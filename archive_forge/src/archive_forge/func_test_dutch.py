import pickle
import hypothesis.strategies as st
import pytest
from hypothesis import given
from spacy import util
from spacy.lang.en import English
from spacy.language import Language
from spacy.pipeline._edit_tree_internals.edit_trees import EditTrees
from spacy.strings import StringStore
from spacy.training import Example
from spacy.util import make_tempdir
def test_dutch():
    strings = StringStore()
    trees = EditTrees(strings)
    tree = trees.add('deelt', 'delen')
    assert trees.tree_to_str(tree) == "(m 0 3 () (m 0 2 (s '' 'l') (s 'lt' 'n')))"
    tree = trees.add('gedeeld', 'delen')
    assert trees.tree_to_str(tree) == "(m 2 3 (s 'ge' '') (m 0 2 (s '' 'l') (s 'ld' 'n')))"