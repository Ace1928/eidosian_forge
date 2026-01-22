from __future__ import annotations
from collections.abc import Iterator
from typing import cast
import pytest
from xarray.core.treenode import InvalidTreeError, NamedNode, NodePath, TreeNode
from xarray.datatree_.datatree.iterators import LevelOrderIter, PreOrderIter
def test_del_child(self):
    john: TreeNode = TreeNode()
    mary: TreeNode = TreeNode()
    john._set_item('Mary', mary)
    del john['Mary']
    assert 'Mary' not in john.children
    assert mary.parent is None
    with pytest.raises(KeyError):
        del john['Mary']