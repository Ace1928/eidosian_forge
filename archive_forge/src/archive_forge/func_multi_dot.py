import numpy as np
import ray
@ray.remote
def multi_dot(*a):
    raise NotImplementedError