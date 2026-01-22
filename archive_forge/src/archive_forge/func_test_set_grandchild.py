from __future__ import annotations
from collections.abc import Iterator
from typing import cast
import pytest
from xarray.core.treenode import InvalidTreeError, NamedNode, NodePath, TreeNode
from xarray.datatree_.datatree.iterators import LevelOrderIter, PreOrderIter
def test_set_grandchild(self):
    rose: TreeNode = TreeNode()
    mary: TreeNode = TreeNode()
    john: TreeNode = TreeNode()
    john._set_item('Mary', mary)
    john._set_item('Mary/Rose', rose)
    assert john.children['Mary'] is mary
    assert isinstance(mary, TreeNode)
    assert 'Rose' in mary.children
    assert rose.parent is mary