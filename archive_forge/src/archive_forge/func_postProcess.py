from __future__ import annotations
from .state_inline import Delimiter, StateInline
def postProcess(state: StateInline) -> None:
    """Walk through delimiter list and replace text tokens with tags."""
    tokens_meta = state.tokens_meta
    maximum = len(state.tokens_meta)
    _postProcess(state, state.delimiters)
    curr = 0
    while curr < maximum:
        try:
            curr_meta = tokens_meta[curr]
        except IndexError:
            pass
        else:
            if curr_meta and 'delimiters' in curr_meta:
                _postProcess(state, curr_meta['delimiters'])
        curr += 1