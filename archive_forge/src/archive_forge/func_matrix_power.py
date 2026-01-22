import numpy as np
import ray
@ray.remote
def matrix_power(M, n):
    return np.linalg.matrix_power(M, n)