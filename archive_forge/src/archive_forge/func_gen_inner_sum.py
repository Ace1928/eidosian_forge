import numpy as np
import functools
import operator
def gen_inner_sum():
    for coo_i in np.ndindex(*inner_size):
        coord.update(dict(zip(inner, coo_i)))
        locs = (tuple((coord[k] for k in term)) for term in inputs)
        elements = (array[loc] for array, loc in zip(arrays, locs))
        yield functools.reduce(operator.mul, elements)