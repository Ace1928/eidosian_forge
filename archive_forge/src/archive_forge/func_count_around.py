import itertools
import functools
import importlib.util
def count_around(c, layout):
    if layout == 'wide':
        yield from itertools.count(c)
    elif layout == 'compact':
        yield from range(c, -1, -1)
        yield from itertools.count(c + 1)
    else:
        step = 0
        sgn = (-1) ** (c <= 0)
        while True:
            cm = c - sgn * step
            if step != 0:
                yield cm
            yield (c + sgn * step)
            step += 1