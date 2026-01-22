from __future__ import annotations
from collections.abc import Iterator
from typing import cast
import pytest
from xarray.core.treenode import InvalidTreeError, NamedNode, NodePath, TreeNode
from xarray.datatree_.datatree.iterators import LevelOrderIter, PreOrderIter
def test_lineage(self):
    _, leaf_f = create_test_tree()
    expected = ['f', 'e', 'b', 'a']
    assert [node.name for node in leaf_f.lineage] == expected