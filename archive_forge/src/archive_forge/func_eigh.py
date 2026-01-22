import numpy as np
import ray
@ray.remote(num_returns=2)
def eigh(a):
    return np.linalg.eigh(a)