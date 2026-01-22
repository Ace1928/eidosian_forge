from collections import OrderedDict, deque
import numpy as np
from ray.rllib.utils import force_list
from ray.rllib.utils.framework import try_import_tf
def unflatten(vector, shapes):
    i = 0
    arrays = []
    for shape in shapes:
        size = np.prod(shape, dtype=np.int_)
        array = vector[i:i + size].reshape(shape)
        arrays.append(array)
        i += size
    assert len(vector) == i, 'Passed weight does not have the correct shape.'
    return arrays