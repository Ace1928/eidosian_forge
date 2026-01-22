import numpy as np
import ray
@ray.remote
def sum_list(*xs):
    return np.sum(xs, axis=0)