import contextlib
import copy
import itertools
import posixpath as pp
import fasteners
from taskflow import exceptions as exc
from taskflow.persistence import path_based
from taskflow.types import tree
def ls_r(self, path, absolute=False):
    """Return list of all children of the given path (recursively)."""
    node = self._fetch_node(path)
    if absolute:
        selector_func = self._metadata_path_selector
    else:
        selector_func = self._up_to_root_selector
    return [selector_func(node, child_node) for child_node in node.bfs_iter()]