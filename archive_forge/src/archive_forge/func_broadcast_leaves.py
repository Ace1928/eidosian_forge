from __future__ import annotations
import difflib
import functools
import itertools
import textwrap
from collections import OrderedDict, defaultdict, deque
from typing import Any, Callable, Iterable, Mapping, overload
from optree import _C
from optree.registry import (
from optree.typing import (
from optree.typing import structseq as PyStructSequence  # noqa: N812
from optree.typing import structseq_fields
def broadcast_leaves(x: T, subtree: PyTree[T]) -> PyTree[T]:
    subtreespec = tree_structure(subtree, is_leaf=is_leaf, none_is_leaf=none_is_leaf, namespace=namespace)
    return subtreespec.unflatten(itertools.repeat(x, subtreespec.num_leaves))