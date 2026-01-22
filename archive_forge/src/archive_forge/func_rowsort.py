import copy
import os
import pickle
import warnings
import numpy as np
def rowsort(self, axis, key=0):
    """Return this object with all records sorted along axis using key as the index to the values to compare. Does not yet modify meta info."""
    keyList = self[key]
    order = keyList.argsort()
    if type(axis) == int:
        ind = [slice(None)] * axis
        ind.append(order)
    elif isinstance(axis, str):
        ind = (slice(axis, order),)
    else:
        raise TypeError('axis must be type (int, str)')
    return self[tuple(ind)]