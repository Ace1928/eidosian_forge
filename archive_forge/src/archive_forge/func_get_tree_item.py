from __future__ import annotations
import abc
import collections
import os
import typing as t
from ...util import (
from .. import (
def get_tree_item(tree: tuple[dict[str, t.Any], list[str]], parts: list[str]) -> t.Optional[tuple[dict[str, t.Any], list[str]]]:
    """Return the portion of the tree found under the path given by parts, or None if it does not exist."""
    root = tree
    for part in parts:
        root = root[0].get(part)
        if not root:
            return None
    return root