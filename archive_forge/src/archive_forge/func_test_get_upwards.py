from __future__ import annotations
from collections.abc import Iterator
from typing import cast
import pytest
from xarray.core.treenode import InvalidTreeError, NamedNode, NodePath, TreeNode
from xarray.datatree_.datatree.iterators import LevelOrderIter, PreOrderIter
def test_get_upwards(self):
    sue: TreeNode = TreeNode()
    kate: TreeNode = TreeNode()
    mary = TreeNode(children={'Sue': sue, 'Kate': kate})
    john = TreeNode(children={'Mary': mary})
    assert sue._get_item('../') is mary
    assert sue._get_item('../../') is john
    assert sue._get_item('../Kate') is kate