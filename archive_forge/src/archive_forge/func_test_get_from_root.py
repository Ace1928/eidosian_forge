from __future__ import annotations
from collections.abc import Iterator
from typing import cast
import pytest
from xarray.core.treenode import InvalidTreeError, NamedNode, NodePath, TreeNode
from xarray.datatree_.datatree.iterators import LevelOrderIter, PreOrderIter
def test_get_from_root(self):
    sue: TreeNode = TreeNode()
    mary = TreeNode(children={'Sue': sue})
    john = TreeNode(children={'Mary': mary})
    assert sue._get_item('/Mary') is mary