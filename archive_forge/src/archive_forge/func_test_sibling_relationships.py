from __future__ import annotations
from collections.abc import Iterator
from typing import cast
import pytest
from xarray.core.treenode import InvalidTreeError, NamedNode, NodePath, TreeNode
from xarray.datatree_.datatree.iterators import LevelOrderIter, PreOrderIter
def test_sibling_relationships(self):
    mary: TreeNode = TreeNode()
    kate: TreeNode = TreeNode()
    ashley: TreeNode = TreeNode()
    TreeNode(children={'Mary': mary, 'Kate': kate, 'Ashley': ashley})
    assert kate.siblings['Mary'] is mary
    assert kate.siblings['Ashley'] is ashley
    assert 'Kate' not in kate.siblings