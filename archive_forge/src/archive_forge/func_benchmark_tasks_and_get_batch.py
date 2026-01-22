import inspect
import logging
import numpy as np
import sys
from ray.util.client.ray_client_helpers import ray_start_client_server
from ray._private.ray_microbenchmark_helpers import timeit
def benchmark_tasks_and_get_batch(ray, results):

    @ray.remote
    def small_value():
        return b'ok'

    def small_value_batch():
        submitted = [small_value.remote() for _ in range(1000)]
        ray.get(submitted)
        return 0
    results += timeit('client: tasks and get batch', small_value_batch)