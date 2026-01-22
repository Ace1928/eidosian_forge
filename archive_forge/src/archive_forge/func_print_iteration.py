from __future__ import annotations
@classmethod
def print_iteration(cls, *args):
    iteration_format = [f'{{:{x}}}' for x in cls.ITERATION_FORMATS]
    fmt = '|' + '|'.join(iteration_format) + '|'
    print(fmt.format(*args))