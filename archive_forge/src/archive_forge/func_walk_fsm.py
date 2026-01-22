from collections import namedtuple
from functools import lru_cache
from typing import TYPE_CHECKING, Dict, Generator, List, Sequence, Set, Tuple
import numba
import numpy as np
from interegular.fsm import FSM, Alphabet, OblivionError, anything_else
from numba.typed.typedobjectutils import _nonoptional
def walk_fsm(fsm: BetterFSM, input_string: str, start_state: int, full_match: bool=True) -> List[int]:
    fsm_finals = fsm.finals
    state = start_state
    accepted_states: List[int] = []
    last_final_idx: int = 0
    alphabet_symbol_mapping = fsm.alphabet._symbol_mapping
    alphabet_anything_value = fsm.alphabet.anything_value
    fsm_transitions = fsm.flat_transition_map
    for i, symbol in enumerate(input_string):
        trans_key = alphabet_symbol_mapping.get(symbol, alphabet_anything_value)
        new_state = fsm_transitions.get((state, trans_key))
        if new_state is None:
            if not full_match and last_final_idx > 0:
                return accepted_states[:last_final_idx]
            return []
        state = new_state
        if state in fsm_finals:
            last_final_idx = i + 1
        accepted_states.append(state)
    if full_match and last_final_idx - 1 != i:
        return []
    return accepted_states