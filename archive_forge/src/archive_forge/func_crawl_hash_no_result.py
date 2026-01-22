from _collections import deque
from collections import defaultdict
from functools import total_ordering
from typing import Any, Set, Dict, Union, NewType, Mapping, Tuple, Iterable
from interegular.utils import soft_repr
def crawl_hash_no_result(alphabet, initial, final, follow):
    unvisited = {initial}
    visited = set()
    while unvisited:
        state = unvisited.pop()
        visited.add(state)
        final(state)
        for transition in alphabet.by_transition:
            try:
                new = follow(state, transition)
            except OblivionError:
                continue
            else:
                if new not in visited:
                    unvisited.add(new)