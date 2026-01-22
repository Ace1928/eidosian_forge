import asyncio
import logging
from ray._private.ray_microbenchmark_helpers import timeit
from ray._private.ray_client_microbenchmark import main as client_microbenchmark_main
import numpy as np
import multiprocessing
import ray
def placement_group_create_removal(num_pgs):
    pgs = [ray.util.placement_group(bundles=[{'custom': 0.001} for _ in range(NUM_BUNDLES)]) for _ in range(num_pgs)]
    [pg.wait(timeout_seconds=30) for pg in pgs]
    for pg in pgs:
        ray.util.remove_placement_group(pg)