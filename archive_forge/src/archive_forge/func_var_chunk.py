from __future__ import annotations
def var_chunk(seq):
    squares, total, n = (0.0, 0.0, 0)
    for x in seq:
        squares += x ** 2
        total += x
        n += 1
    return (squares, total, n)