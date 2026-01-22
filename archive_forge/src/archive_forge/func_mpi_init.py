import logging
import os
from typing import List, Optional
from ray._private.runtime_env.context import RuntimeEnvContext
from ray._private.runtime_env.plugin import RuntimeEnvPlugin
import subprocess
def mpi_init():
    """Initialize the MPI cluster. When using MPI cluster, this must be called first."""
    if hasattr(mpi_init, 'inited'):
        assert mpi_init.inited is True
        return
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        from ray._private.accelerators import get_all_accelerator_managers
        device_vars = [m.get_visible_accelerator_ids_env_var() for m in get_all_accelerator_managers()]
        visible_devices = {n: os.environ.get(n) for n in device_vars if os.environ.get(n)}
        comm.bcast(visible_devices)
        with open(f'/tmp/{os.getpid()}.{rank}', 'w') as f:
            f.write(str(visible_devices))
    else:
        visible_devices = comm.bcast(None)
        os.environ.update(visible_devices)
    mpi_init.inited = True