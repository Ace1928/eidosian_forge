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
def test_from_to_disk():
    strings = StringStore()
    trees = EditTrees(strings)
    trees.add('deelt', 'delen')
    trees.add('gedeeld', 'delen')
    trees2 = EditTrees(strings)
    with make_tempdir() as temp_dir:
        trees_file = temp_dir / 'edit_trees.bin'
        trees.to_disk(trees_file)
        trees2 = trees2.from_disk(trees_file)
    assert len(trees) == len(trees2)
    for i in range(len(trees)):
        assert trees.tree_to_str(i) == trees2.tree_to_str(i)
    trees2.add('deelt', 'delen')
    trees2.add('gedeeld', 'delen')
    assert len(trees) == len(trees2)