import io
import math
import os
import typing
import weakref
def output_justify(start, line):
    """Justified output of a line."""
    words = [w for w in line.split(' ') if w != '']
    nwords = len(words)
    if nwords == 0:
        return
    if nwords == 1:
        append_this(start, words[0])
        return
    tl = sum([textlen(w) for w in words])
    gaps = nwords - 1
    gapl = (std_width - tl) / gaps
    for w in words:
        _, lp = append_this(start, w)
        start.x = lp.x + gapl
    return