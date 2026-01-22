from _collections import deque
from collections import defaultdict
from functools import total_ordering
from typing import Any, Set, Dict, Union, NewType, Mapping, Tuple, Iterable
from interegular.utils import soft_repr
def islive(self, state):
    """A state is "live" if a final state can be reached from it."""
    seen = {state}
    reachable = [state]
    i = 0
    while i < len(reachable):
        current = reachable[i]
        if current in self.finals:
            return True
        if current in self.map:
            for transition in self.map[current]:
                next = self.map[current][transition]
                if next not in seen:
                    reachable.append(next)
                    seen.add(next)
        i += 1
    return False