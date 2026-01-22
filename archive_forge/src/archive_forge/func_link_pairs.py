from __future__ import annotations
from .state_inline import Delimiter, StateInline
def link_pairs(state: StateInline) -> None:
    tokens_meta = state.tokens_meta
    maximum = len(state.tokens_meta)
    processDelimiters(state, state.delimiters)
    curr = 0
    while curr < maximum:
        curr_meta = tokens_meta[curr]
        if curr_meta and 'delimiters' in curr_meta:
            processDelimiters(state, curr_meta['delimiters'])
        curr += 1