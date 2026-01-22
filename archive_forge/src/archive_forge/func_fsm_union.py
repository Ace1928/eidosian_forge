from collections import namedtuple
from functools import lru_cache
from typing import TYPE_CHECKING, Dict, Generator, List, Sequence, Set, Tuple
import numba
import numpy as np
from interegular.fsm import FSM, Alphabet, OblivionError, anything_else
from numba.typed.typedobjectutils import _nonoptional
def fsm_union(fsms: Sequence[FSM]) -> Tuple[FSM, Dict[int, Tuple[Set[Tuple[int, int]], Set[int], Dict[int, Set[int]]]]]:
    """Construct an FSM representing the union of the FSMs in `fsms`.

    This is an updated version of `interegular.fsm.FSM.union` made to return an
    extra map of component FSMs to the sets of state transitions that
    correspond to them in the new FSM.

    """
    alphabet, new_to_old = Alphabet.union(*[fsm.alphabet for fsm in fsms])
    indexed_fsms = tuple(enumerate(fsms))
    initial = {i: fsm.initial for i, fsm in indexed_fsms}

    def follow(current_state, new_transition: int):
        next = {}
        for i, f in indexed_fsms:
            old_transition = new_to_old[i][new_transition]
            if i in current_state and current_state[i] in f.map and (old_transition in f.map[current_state[i]]):
                next[i] = f.map[current_state[i]][old_transition]
        if not next:
            raise OblivionError
        return next
    states = [initial]
    finals: Set[int] = set()
    map: Dict[int, Dict[int, int]] = {}
    fsms_to_trans_finals: Dict[int, Tuple[Set[Tuple[int, int]], Set[int], Dict[int, Set[int]]]] = {}
    i = 0
    while i < len(states):
        state = states[i]
        if any((state.get(j, -1) in fsm.finals for j, fsm in indexed_fsms)):
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
                for fsm_id, fsm_state in next.items():
                    fsm_transitions, fsm_finals, fsm_old_to_new = fsms_to_trans_finals.setdefault(fsm_id, (set(), set(), {}))
                    old_from = state[fsm_id]
                    old_to = fsm_state
                    fsm_old_to_new.setdefault(old_from, set()).add(i)
                    fsm_old_to_new.setdefault(old_to, set()).add(j)
                    fsm_transitions.add((i, j))
                    if fsm_state in fsms[fsm_id].finals:
                        fsm_finals.add(j)
        i += 1
    fsm = FSM(alphabet=alphabet, states=range(len(states)), initial=0, finals=finals, map=map, __no_validation__=True)
    fsm, old_to_new_states = make_deterministic_fsm(fsm)
    _fsms_to_trans_finals = {fsm_id: ({(old_to_new_states[s1], old_to_new_states[s2]) for s1, s2 in transitions}, {old_to_new_states[s] for s in finals}, {old_state: {old_to_new_states[new_state] for new_state in new_states} for old_state, new_states in old_to_new.items()}) for fsm_id, (transitions, finals, old_to_new) in sorted(fsms_to_trans_finals.items(), key=lambda x: x[0])}
    return (fsm, _fsms_to_trans_finals)