from ray.util.client import ray
from typing import Tuple
@ray.remote
def plus2(x):
    return x + 2