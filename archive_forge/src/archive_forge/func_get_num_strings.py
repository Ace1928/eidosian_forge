from _collections import deque
from collections import defaultdict
from functools import total_ordering
from typing import Any, Set, Dict, Union, NewType, Mapping, Tuple, Iterable
from interegular.utils import soft_repr
def get_num_strings(state):
    if self.islive(state):
        if state in num_strings:
            if num_strings[state] is None:
                raise OverflowError(state)
            return num_strings[state]
        num_strings[state] = None
        n = 0
        if state in self.finals:
            n += 1
        if state in self.map:
            for transition in self.map[state]:
                n += get_num_strings(self.map[state][transition]) * len(self.alphabet.by_transition[transition])
        num_strings[state] = n
    else:
        num_strings[state] = 0
    return num_strings[state]