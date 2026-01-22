from __future__ import annotations
import copy
from collections import defaultdict, deque, namedtuple
from collections.abc import Iterable, Mapping, MutableMapping
from typing import Any, Callable, Literal, NamedTuple, overload
from dask.core import get_dependencies, get_deps, getcycle, istask, reverse_dict
from dask.typing import Key
def use_longest_path() -> bool:
    size = 0
    if abs(len(root_nodes) - len(leaf_nodes)) / len(root_nodes) < 0.8:
        for r in root_nodes:
            if not size:
                size = len(leafs_connected[r])
            elif size != len(leafs_connected[r]):
                return False
    return True