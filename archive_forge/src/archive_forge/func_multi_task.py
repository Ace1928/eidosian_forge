import asyncio
import logging
from ray._private.ray_microbenchmark_helpers import timeit
from ray._private.ray_client_microbenchmark import main as client_microbenchmark_main
import numpy as np
import multiprocessing
import ray
def multi_task():
    submitted = [a.small_value_batch.remote(n) for a in actors]
    ray.get(submitted)