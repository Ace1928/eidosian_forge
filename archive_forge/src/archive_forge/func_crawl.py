from _collections import deque
from collections import defaultdict
from functools import total_ordering
from typing import Any, Set, Dict, Union, NewType, Mapping, Tuple, Iterable
from interegular.utils import soft_repr
def crawl(alphabet, initial, final, follow):
    """
        Given the above conditions and instructions, crawl a new unknown FSM,
        mapping its states, final states and transitions. Return the new FSM.
        This is a pretty powerful procedure which could potentially go on
        forever if you supply an evil version of follow().
    """
    states = [initial]
    finals = set()
    map = {}
    i = 0
    while i < len(states):
        state = states[i]
        if final(state):
            finals.add(i)
        map[i] = {}
        for transition in alphabet.by_transition:
            try:
                next = follow(state, transition)
            except OblivionError:
                continue
            else:
                try:
                    j = states.index(next)
                except ValueError:
                    j = len(states)
                    states.append(next)
                map[i][transition] = j
        i += 1
    return FSM(alphabet=alphabet, states=range(len(states)), initial=0, finals=finals, map=map, __no_validation__=True)