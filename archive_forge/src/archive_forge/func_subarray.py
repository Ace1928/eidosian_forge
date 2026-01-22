import numpy as np
import ray
@ray.remote
def subarray(a, lower_indices, upper_indices):
    idx = tuple((slice(l, u) for l, u in zip(lower_indices, upper_indices)))
    return a[idx]