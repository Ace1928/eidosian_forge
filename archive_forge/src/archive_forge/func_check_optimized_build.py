import asyncio
import logging
from ray._private.ray_microbenchmark_helpers import timeit
from ray._private.ray_client_microbenchmark import main as client_microbenchmark_main
import numpy as np
import multiprocessing
import ray
def check_optimized_build():
    if not ray._raylet.OPTIMIZED:
        msg = 'WARNING: Unoptimized build! To benchmark an optimized build, try:\n\tbazel build -c opt //:ray_pkg\nYou can also make this permanent by adding\n\tbuild --compilation_mode=opt\nto your user-wide ~/.bazelrc file. (Do not add this to the project-level .bazelrc file.)'
        logger.warning(msg)