from __future__ import annotations
from collections.abc import Iterator
from typing import cast
import pytest
from xarray.core.treenode import InvalidTreeError, NamedNode, NodePath, TreeNode
from xarray.datatree_.datatree.iterators import LevelOrderIter, PreOrderIter
def test_doppelganger_child(self):
    kate: TreeNode = TreeNode()
    john: TreeNode = TreeNode()
    with pytest.raises(TypeError):
        john.children = {'Kate': 666}
    with pytest.raises(InvalidTreeError, match='Cannot add same node'):
        john.children = {'Kate': kate, 'Evil_Kate': kate}
    john = TreeNode(children={'Kate': kate})
    evil_kate: TreeNode = TreeNode()
    evil_kate._set_parent(john, 'Kate')
    assert john.children['Kate'] is evil_kate