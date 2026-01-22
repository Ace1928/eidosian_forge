import logging
from ..common.utils import isStrSpace
from .state_block import StateBlock
def skipOrderedListMarker(state: StateBlock, startLine: int) -> int:
    start = state.bMarks[startLine] + state.tShift[startLine]
    pos = start
    maximum = state.eMarks[startLine]
    if pos + 1 >= maximum:
        return -1
    ch = state.src[pos]
    pos += 1
    ch_ord = ord(ch)
    if ch_ord < 48 or ch_ord > 57:
        return -1
    while True:
        if pos >= maximum:
            return -1
        ch = state.src[pos]
        pos += 1
        ch_ord = ord(ch)
        if ch_ord >= 48 and ch_ord <= 57:
            if pos - start >= 10:
                return -1
            continue
        if ch in (')', '.'):
            break
        return -1
    if pos < maximum:
        ch = state.src[pos]
        if not isStrSpace(ch):
            return -1
    return pos