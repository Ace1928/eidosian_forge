import numpy as np
import ray
@ray.remote(num_returns=2)
def tensorinv(a):
    raise NotImplementedError