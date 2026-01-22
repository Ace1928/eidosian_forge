import logging
import os
from typing import List, Optional
from ray._private.runtime_env.context import RuntimeEnvContext
from ray._private.runtime_env.plugin import RuntimeEnvPlugin
import subprocess
This plugin enable a MPI cluster to run on top of ray.

    To use this, "mpi" need to be added to the runtime env like following

    @ray.remote(
        runtime_env={
            "mpi": {
                "args": ["-n", "4"],
                "worker_entry": worker_entry,
            }
        }
    )
    def calc_pi():
      ...

    Here worker_entry should be function for the MPI worker to run.
    For example, it should be `'py_module.worker_func'`. The module should be able to
    be imported in the runtime.

    In the mpi worker with rank==0, it'll be the normal ray function or actor.
    For the worker with rank > 0, it'll just run `worker_func`.

    ray.runtime_env.mpi_init must be called in the ray actors/tasks before any MPI
    communication.
    