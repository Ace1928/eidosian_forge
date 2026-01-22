import contextlib
import copy
import itertools
import posixpath as pp
import fasteners
from taskflow import exceptions as exc
from taskflow.persistence import path_based
from taskflow.types import tree
def ensure_path(self, path):
    """Ensure the path (and parents) exists."""
    path = self.normpath(path)
    if path == self._root.item:
        return
    node = self._root
    for piece in self._iter_pieces(path):
        child_node = node.find(piece, only_direct=True, include_self=False)
        if child_node is None:
            child_node = self._insert_child(node, piece)
        node = child_node