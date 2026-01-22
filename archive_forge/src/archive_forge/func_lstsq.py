import numpy as np
import ray
@ray.remote(num_returns=4)
def lstsq(a, b):
    return np.linalg.lstsq(a)