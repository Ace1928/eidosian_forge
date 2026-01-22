import asyncio
import logging
from ray._private.ray_microbenchmark_helpers import timeit
from ray._private.ray_client_microbenchmark import main as client_microbenchmark_main
import numpy as np
import multiprocessing
import ray
@ray.remote
def work(actors):
    ray.get([actors[i % n_cpu].small_value.remote() for i in range(n)])