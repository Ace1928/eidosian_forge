from _collections import deque
from collections import defaultdict
from functools import total_ordering
from typing import Any, Set, Dict, Union, NewType, Mapping, Tuple, Iterable
from interegular.utils import soft_repr
def everythingbut(self):
    """
            Return a finite state machine which will accept any string NOT
            accepted by self, and will not accept any string accepted by self.
            This is more complicated if there are missing transitions, because the
            missing "dead" state must now be reified.
        """
    alphabet = self.alphabet
    initial = {0: self.initial}

    def follow(current, transition):
        next = {}
        if 0 in current and current[0] in self.map and (transition in self.map[current[0]]):
            next[0] = self.map[current[0]][transition]
        return next

    def final(state):
        return not (0 in state and state[0] in self.finals)
    return crawl(alphabet, initial, final, follow)