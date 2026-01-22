from _collections import deque
from collections import defaultdict
from functools import total_ordering
from typing import Any, Set, Dict, Union, NewType, Mapping, Tuple, Iterable
from interegular.utils import soft_repr
def parallel(fsms, test):
    """
        Crawl several FSMs in parallel, mapping the states of a larger meta-FSM.
        To determine whether a state in the larger FSM is final, pass all of the
        finality statuses (e.g. [True, False, False] to `test`.
    """
    alphabet, new_to_old = Alphabet.union(*[fsm.alphabet for fsm in fsms])
    initial = {i: fsm.initial for i, fsm in enumerate(fsms)}

    def follow(current, new_transition, fsm_range=tuple(enumerate(fsms))):
        next = {}
        for i, f in fsm_range:
            old_transition = new_to_old[i][new_transition]
            if i in current and current[i] in f.map and (old_transition in f.map[current[i]]):
                next[i] = f.map[current[i]][old_transition]
        if not next:
            raise OblivionError
        return next

    def final(state, fsm_range=tuple(enumerate(fsms))):
        accepts = [i in state and state[i] in fsm.finals for i, fsm in fsm_range]
        return test(accepts)
    return crawl(alphabet, initial, final, follow)