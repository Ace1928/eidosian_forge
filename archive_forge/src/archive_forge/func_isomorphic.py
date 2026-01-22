from __future__ import annotations
import copy
import itertools
from collections.abc import Hashable, Iterable, Iterator, Mapping, MutableMapping
from html import escape
from typing import (
from xarray.core import utils
from xarray.core.coordinates import DatasetCoordinates
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset, DataVariables
from xarray.core.indexes import Index, Indexes
from xarray.core.merge import dataset_update_method
from xarray.core.options import OPTIONS as XR_OPTS
from xarray.core.treenode import NamedNode, NodePath, Tree
from xarray.core.utils import (
from xarray.core.variable import Variable
from xarray.datatree_.datatree.common import TreeAttrAccessMixin
from xarray.datatree_.datatree.formatting import datatree_repr
from xarray.datatree_.datatree.formatting_html import (
from xarray.datatree_.datatree.mapping import (
from xarray.datatree_.datatree.ops import (
from xarray.datatree_.datatree.render import RenderTree
def isomorphic(self, other: DataTree, from_root: bool=False, strict_names: bool=False) -> bool:
    """
        Two DataTrees are considered isomorphic if every node has the same number of children.

        Nothing about the data in each node is checked.

        Isomorphism is a necessary condition for two trees to be used in a nodewise binary operation,
        such as ``tree1 + tree2``.

        By default this method does not check any part of the tree above the given node.
        Therefore this method can be used as default to check that two subtrees are isomorphic.

        Parameters
        ----------
        other : DataTree
            The other tree object to compare to.
        from_root : bool, optional, default is False
            Whether or not to first traverse to the root of the two trees before checking for isomorphism.
            If neither tree has a parent then this has no effect.
        strict_names : bool, optional, default is False
            Whether or not to also check that every node in the tree has the same name as its counterpart in the other
            tree.

        See Also
        --------
        DataTree.equals
        DataTree.identical
        """
    try:
        check_isomorphic(self, other, require_names_equal=strict_names, check_from_root=from_root)
        return True
    except (TypeError, TreeIsomorphismError):
        return False