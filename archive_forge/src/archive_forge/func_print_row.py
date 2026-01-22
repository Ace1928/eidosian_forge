import functools
import weakref
import numpy as np
from tensorflow.python.util import nest
def print_row(fields, positions):
    line = ''
    for i in range(len(fields)):
        if i > 0:
            line = line[:-1] + ' '
        line += str(fields[i])
        line = line[:positions[i]]
        line += ' ' * (positions[i] - len(line))
    print_fn(line)