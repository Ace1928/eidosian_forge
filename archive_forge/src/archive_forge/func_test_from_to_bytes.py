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
def test_from_to_bytes():
    strings = StringStore()
    trees = EditTrees(strings)
    trees.add('deelt', 'delen')
    trees.add('gedeeld', 'delen')
    b = trees.to_bytes()
    trees2 = EditTrees(strings)
    trees2.from_bytes(b)
    assert len(trees) == len(trees2)
    for i in range(len(trees)):
        assert trees.tree_to_str(i) == trees2.tree_to_str(i)
    trees2.add('deelt', 'delen')
    trees2.add('gedeeld', 'delen')
    assert len(trees) == len(trees2)